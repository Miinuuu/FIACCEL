"""Independent scalar integer oracles; no RTL or simulator dependency."""

def reference(image, w, cout, k=3, stride=1, dilation=1, transpose=False,
              shift=3, head=False):
    """HWC arrays. Transpose uses scatter, independent of RTL's gather."""
    h, width, cin = len(image), len(image[0]), len(image[0][0])
    oh = 2*h if transpose else (h+stride-1)//stride
    ow = 2*width if transpose else (width+stride-1)//stride
    out = [[[0]*cout for _ in range(ow)] for _ in range(oh)]
    if transpose:
        for iy in range(h):
            for ix in range(width):
                for ci in range(cin):
                    for co in range(cout):
                        for ky in range(k):
                            for kx in range(k):
                                y, x = 2*iy+ky-1, 2*ix+kx-1
                                if 0 <= y < oh and 0 <= x < ow:
                                    out[y][x][co] += image[iy][ix][ci]*w[((co*cin+ci)*k+ky)*k+kx]
    else:
        for y in range(oh):
            for x in range(ow):
                for co in range(cout):
                    total = 0
                    for ci in range(cin):
                        for ky in range(k):
                            for kx in range(k):
                                iy, ix = y*stride+ky-k//2, x*stride+(kx-k//2)*dilation
                                if 0 <= iy < h and 0 <= ix < width:
                                    total += image[iy][ix][ci]*w[((co*cin+ci)*k+ky)*k+kx]
                    out[y][x][co] = total
    lo, hi = (-32768, 32767) if head else (0, 255)
    return [[[max(lo, min(hi, v >> shift)) for v in p] for p in row] for row in out]

def synthesize(previous,current,voxel):
    h,w=len(previous),len(previous[0]);out=[]
    def sample(img,xq,yq):
        xq=max(0,min((w-1)*256,xq));yq=max(0,min((h-1)*256,yq))
        x0,xf=divmod(xq,256);y0,yf=divmod(yq,256)
        x1=min(w-1,x0+1);y1=min(h-1,y0+1)
        return [img[y0][x0][c]*(256-xf)*(256-yf)+img[y0][x1][c]*xf*(256-yf)+
                img[y1][x0][c]*(256-xf)*yf+img[y1][x1][c]*xf*yf for c in range(3)]
    for y in range(h):
        row=[]
        for x in range(w):
            dx,dy,mask=voxel[y][x]
            sx=(dx*(w-1))>>8;sy=(dy*(h-1))>>8
            a=sample(previous,x*256-sx,y*256-sy);b=sample(current,x*256+sx,y*256+sy)
            alpha=16384+mask
            row.append([(u*alpha+v*(32768-alpha)+(1<<30))>>31 for u,v in zip(a,b)])
        out.append(row)
    return out
