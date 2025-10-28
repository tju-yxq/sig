#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 初始化全局张量
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        
        // 保存参数
        this->totalLength = totalLength;
        this->tileNum = tileNum;
        
        // 计算每个tile的大小
        this->tileLength = totalLength / tileNum;
        this->tailLength = totalLength % tileNum;
        
        // 计算buffer大小，需要确保至少容纳一个tile的数据
        uint32_t bufferSize = (this->tileLength > 0) ? this->tileLength : totalLength;
        
        // 初始化管道
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufferSize * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, bufferSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        // 计算循环次数，处理tiling
        int32_t loopCount = this->tileNum;
        if (this->tailLength > 0) {
            loopCount += 1;
        }
        
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 计算当前tile的数据长度
        uint32_t currentLength = (progress < this->tileNum) ? this->tileLength : this->tailLength;
        if (currentLength == 0) return;
        
        // 分配队列
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        
        // 从全局内存拷贝数据到本地内存
        DataCopy(xLocal, xGm[progress * this->tileLength], currentLength);
        
        // 释放队列
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 计算当前tile的数据长度
        uint32_t currentLength = (progress < this->tileNum) ? this->tileLength : this->tailLength;
        if (currentLength == 0) return;
        
        // 获取输入数据
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        // 执行sigmoid计算: y = 1 / (1 + exp(-x))
        // 使用AscendC提供的Sigmoid函数
        Sigmoid(yLocal, xLocal, currentLength);
        
        // 释放队列
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 计算当前tile的数据长度
        uint32_t currentLength = (progress < this->tileNum) ? this->tileLength : this->tailLength;
        if (currentLength == 0) return;
        
        // 获取计算结果
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        
        // 从本地内存拷贝数据到全局内存
        DataCopy(yGm[progress * this->tileLength], yLocal, currentLength);
        
        // 释放队列
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailLength;

};
extern "C" __global__ __aicore__ void sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    // 创建算子实例
    KernelSigmoid op;
    
    // 初始化算子
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    
    // 执行计算
    op.Process();
}