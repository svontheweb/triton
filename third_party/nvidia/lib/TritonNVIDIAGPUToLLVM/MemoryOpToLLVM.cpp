#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (auto config = chooseLoadMatrixConfig(srcTy, dstTy)) {
      applyMatrixConfig(op, adaptor, *config, rewriter);
      return success();
    }
    return failure();
  }

private:
  void applyMatrixConfig(triton::gpu::LocalLoadOp op,
                         triton::gpu::LocalLoadOpAdaptor adaptor,
                         const LoadStoreMatrixConfig &config,
                         ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dstTy = op.getType();

    auto bitWidth = dstTy.getElementTypeBitWidth();
    auto dstEnc = dstTy.getEncoding();
    auto shape = dstTy.getShape();
    auto dstLL = toLinearLayout(shape, dstEnc);

    auto typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto kLane = str_attr("lane");
    auto withCTAOffset = getNumCTAs(dstEnc) > 1;
    auto hardwareTuple = emitHardwareTuple(
        loc, rewriter, targetInfo, withCTAOffset, dstLL.getInDimSize(kLane));
    auto [laneId, warpId, blockId] = hardwareTuple;

    // Emit ldmatrix load operations for values packed in i32s
    SmallVector<Value> elemsI32;
    auto numTiles = product<unsigned>(config.numTiles);
    auto regVec = numTiles * 32 / bitWidth;
    auto numElemsI32PerTile = (regVec * bitWidth / 32);

    auto matTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(numTiles, i32_ty));
    auto i8 = b.i32_val(8);
    auto i4 = b.i32_val(4);
    auto targetLaneId =
        b.add(b.mul(b.udiv(laneId, i8), i8), b.mul(b.urem(laneId, i4), i4));
    std::tuple<Value, Value, Value> targetHardwareTuple = {targetLaneId, warpId,
                                                           blockId};

    emitTransferBetweenRegistersAndShared(
        dstTy, srcTy, llvmElemTy, regVec, smemObj, loc, rewriter, targetInfo,
        targetHardwareTuple, [&](VectorType vecTy, Value vecAddr) {
          auto ldMatrixOp = rewriter.create<nvgpu::LoadMatrixOp>(
              loc, matTy, vecAddr, /*needTrans=*/config.trans);
          auto res = ldMatrixOp.getResult();
          for (int i = 0; i < numElemsI32PerTile; ++i)
            elemsI32.push_back(b.extract_val(i32_ty, res, i));
        });

    // Unpack i32 values to the original type
    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / bitWidth;
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = b.bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), llvmElemTy));
    auto ret = packLLElements(loc, typeConverter, elems, rewriter, structTy);
    rewriter.replaceOp(op, ret);
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

LogicalResult lowerDistributedToSharedStmatrix(
    Location loc, TypedValue<RankedTensorType> src, MemDescType memDescType,
    Value adaptorSrc, Value smemBase, const TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto mmaEncoding =
      dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(src.getType().getEncoding());
  if (!mmaEncoding)
    return failure();
  auto sharedLayout =
      cast<triton::gpu::SharedEncodingAttr>(memDescType.getEncoding());
  if (!sharedLayout.getHasLeadingOffset())
    return failure();
  int swizzleByteSize = 0;
  if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
    swizzleByteSize = 32;
  else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
    swizzleByteSize = 64;
  else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
    swizzleByteSize = 128;
  else
    return failure();

  RankedTensorType srcTy = src.getType();
  SmallVector<unsigned> shape =
      convertType<unsigned, int64_t>(srcTy.getShape());
  auto order = sharedLayout.getOrder();
  if (!targetInfo.canUseStMatrix(srcTy, shape, shape, order, swizzleByteSize)) {
    return failure();
  }

  auto *ctx = rewriter.getContext();

  auto layout =
      chooseStMatrixLayout(rewriter.getContext(), srcTy, swizzleByteSize);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  auto smemPtrTy = ptr_ty(ctx, 3);

  auto kRegister = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = b.i32_val(layout.getInDimSize(kLane));
  Value laneId = b.urem(threadId, threadsPerWarp);
  Value warpId = b.udiv(threadId, threadsPerWarp);

  auto regBase = applyLinearLayout(loc, rewriter, layout,
                                   {{kRegister, b.i32_val(0)},
                                    {kLane, laneId},
                                    {kWarp, warpId},
                                    {kBlock, b.i32_val(0)}})[0]
                     .second;
  auto srcVals = unpackLLElements(loc, adaptorSrc, rewriter);
  auto srcVec = layout.getNumConsecutiveInOut();
  for (int i = 0; i < srcVals.size(); i += srcVec) {
    auto regIdx =
        layout.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})[0]
            .second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset);
    vecAddr.setInbounds(true);
    SmallVector<Value> inValsVec;
    for (int j = 0; j < srcVec; j++)
      inValsVec.push_back(srcVals[i + j]);
    Value valsVec = packLLVector(loc, inValsVec, rewriter);
    targetInfo.storeMatrixShared(rewriter, loc, vecAddr, valsVec);
  }
  return success();
}

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(memDescType.getEncoding());
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);

    if (lowerDistributedToSharedStmatrix(op.getLoc(), op.getSrc(), memDescType,
                                         adaptor.getSrc(), smemBase,
                                         typeConverter, rewriter, targetInfo)
            .failed()) {
      return failure();
    }

    auto resultTy = cast<MemDescType>(op.getType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, resultTy.getRank(),
                                      op.getLoc(), rewriter);
    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    MemDescType memDescType = op.getDst().getType();
    if (lowerDistributedToSharedStmatrix(
            op.getLoc(), op.getSrc(), memDescType, adaptor.getSrc(),
            smemObj.getBase(), getTypeConverter(), rewriter, targetInfo)
            .failed()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
