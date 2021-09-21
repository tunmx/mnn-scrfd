//
// Created by tunm on 2021/9/21.
//

#ifndef MNN_SCRFD_MODEL_LOAD_H
#define MNN_SCRFD_MODEL_LOAD_H
#include <stdio.h>
#include <libgen.h>


typedef struct DetHeadInfo {
    std::string cls_layer;
    std::string box_layer;
    std::string lmk_layer;
    int stride;
} DetHeadInfo;

static std::vector<DetHeadInfo> scrfd_2_5g_bnkps_head_info{
        {"446", "449", "452", 8},
        {"466", "469", "472", 16},
        {"486", "489", "492", 32},
};
static std::vector<DetHeadInfo> scrfd_500m_bnkps_head_info{
        {"443", "446", "449", 8},
        {"468", "471", "474", 16},
        {"493", "496", "499", 32},
};



#endif //MNN_SCRFD_MODEL_LOAD_H
