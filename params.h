//
// Created by mark on 2024-05-15.
//

#ifndef MINI_FLASHATTENTION_PARAMS_H
#define MINI_FLASHATTENTION_PARAMS_H

#endif //MINI_FLASHATTENTION_PARAMS_H
struct Tile{
    int m;
    int n;
    int k;
    int warps_m;
    int warps_n;
    int warps_k;
};

struct Params{
    int b;
    int s;
    int h;
    int d;
    Tile tile_q;
    Tile tile_k;
    Tile tile_v;

};