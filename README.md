# Radial voting
nonofficial implementation of radial voting methods.



# Dependency 

`opencv` `numpy`

# Details

This repository:

> rmin = 0.5 * mean cell radius
>
> rmax = 1.5 * mean cell radius
>
> delta of angle = 30
>
> meanshift bandwidth = 1/3 * mean cell diameter
>
> speed = 10s per image with size [300*300] 

Original in paper:

> rmin = 0.5 * mean cell diameter
>
> rmax = 1.5 * mean cell diameter
>
> delta of angle = 30
>
> meanshift bandwidth = 1/3 * mean cell diameter
>
> speed(gpu) = 0.2s per image with size [1000*1000]


# Reference
* [iterative radial voting](https://ieeexplore.ieee.org/abstract/document/4099402)
* [single_pass voting](https://ieeexplore.ieee.org/abstract/document/6099601)
* [heirarchical voting](https://ieeexplore.ieee.org/abstract/document/6670688)
