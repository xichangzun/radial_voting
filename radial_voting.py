import numpy as np
import cv2
import math
import time
from multiprocessing import Process,Manager,Pool

def default_weight(*args):
    return 1

def gauss_weight(y,x,center_y,center_x,sigma):
    s = 2*(sigma)**2
    index = -1*((x-center_x)**2+(y-center_y)**2)/s
    weight = math.exp(index)/(s*math.pi)
    return weight

_delta = 30*math.pi/180

def get_area_templates(rmin,rmax,d=_delta,weight=default_weight):
    area_templates = []
    num = 360
    if rmax < 30:
        num = 180
    elif rmax < 15:
        num = 90
    
    radio = 2*math.pi/num
    for i in range(0,num):
        init_area = np.zeros((rmax*2+1,rmax*2+1))
        orient = i*radio
        y = rmax
        x = rmax
        center_y = math.floor(y+((rmin+rmax)/2)*math.sin(orient))
        center_x = math.floor(x+((rmin+rmax)/2)*math.cos(orient))
        for p in range(0,2*rmax+1):
            for q in range(0,2*rmax+1):
                dist = (p-x)**2+(q-y)**2
                if dist > rmin**2 and dist < rmax**2 :
                    vec_angle = math.atan2(q-y,p-x)
                    if vec_angle < 0:
                        vec_angle += math.pi*2
                    angle_diff = abs(vec_angle - orient)
                    if  angle_diff <= d  or angle_diff >= 2*math.pi-d:
                        init_area[q][p] = 1*weight(q,p,center_y,center_x,(rmax-rmin)/2)
                else:
                    continue
        
        area_templates.append(init_area)
    
    return area_templates


def v_a_from_templates(y,x,area_templates,angle):
    orient = (angle[y][x]+math.pi)%(2*math.pi)
    radio =  2*math.pi/len(area_templates)
    index = math.floor((orient/radio)+0.5)
    index = index%len(area_templates) #radius [359.5,360)
    voting_area = np.zeros_like(angle)
    sample = area_templates[index].copy()
    _,lenth = sample.shape
    rmax = (lenth-1)//2
    move_y = rmax-y
    move_x = rmax-x
    
    x_low = max(x-rmax,0)
    x_high = min(x+rmax,w)
    y_low = max(y-rmax,0)
    y_high = min(y+rmax,h)

    s_x_l = x_low + move_x
    s_x_h = x_high + move_x
    s_y_l = y_low + move_y
    s_y_h = y_high + move_y


    voting_area[y_low:y_high,x_low:x_high] = sample[s_y_l:s_y_h,s_x_l:s_x_h]

    return voting_area
    


def mean_shift(src,bandwidth):
    flags = np.zeros_like(src)
    h,w = src.shape

    def get_center(cy,cx):
        xl = max(0,cx-bandwidth)
        xh = min(w,cx+bandwidth)
        yl = max(0,cy-bandwidth)
        yh = min(h,cy+bandwidth)
        count = 0
        x_sum = 0
        y_sum = 0
        for p in range(xl,xh):
            for q in range(yl,yh):
                num = src[q][p]
                count += num
                x_sum += num*p
                y_sum += num*q
        if count == 0:
            return cy,cx,count
        else:
            mean_y = math.floor(y_sum / count + 0.5)
            mean_x = math.floor(x_sum / count + 0.5)
            return mean_y, mean_x,count
    
    seeds = []
    
    for i in range(w):
        for j in range(h):
            if flags[j][i] == 1 :
                continue
            x = i
            y = j
            while True:
                y_m,x_m,c = get_center(y,x)
                if flags[y_m][x_m] == 1:
                    break
                if y_m == y and x_m == x :
                    flags[y_m][x_m] =1
                    if c > 0:
                        seeds.append((y_m,x_m))
                    break
                y = y_m
                x = x_m

    return seeds


