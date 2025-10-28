/**
* @file aclnn_sigmoid_custom.h
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef ACLNN_SIGMOID_CUSTOM_H_
#define ACLNN_SIGMOID_CUSTOM_H_

#include "acl/acl.h"
#include "acl/acl_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSigmoidCustomGetWorkspaceSize interface
 * @param [in] x: input tensor
 * @param [in] out: output tensor  
 * @param [out] workspaceSize: size of workspace
 * @param [out] executor: pointer of aclOpExecutor
 * @return aclError
 */
aclError aclnnSigmoidCustomGetWorkspaceSize(const aclTensor *x,
                                          const aclTensor *out,
                                          uint64_t *workspaceSize,
                                          aclOpExecutor **executor);

/**
 * @brief aclnnSigmoidCustom interface
 * @param [in] workspace: workspace for the operator
 * @param [in] workspaceSize: size of workspace
 * @param [in] executor: pointer of aclOpExecutor  
 * @param [in] stream: acl stream
 * @return aclError
 */
aclError aclnnSigmoidCustom(void *workspace,
                          uint64_t workspaceSize,
                          aclOpExecutor *executor,
                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SIGMOID_CUSTOM_H_