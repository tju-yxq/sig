
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SigmoidCustomTilingData)
//考生自行定义tiling结构体成员变量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SigmoidCustom, SigmoidCustomTilingData)
}
