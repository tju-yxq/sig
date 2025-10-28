
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SigmoidCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, BLOCK_DIM);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SigmoidCustom, SigmoidCustomTilingData)
}
