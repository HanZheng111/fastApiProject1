def GetResizeFromWidth(size ,width):
    high=size[1]*width/size[0]
    return [width,high]

def GetResizeFromHigh(size ,high):
    width=size[0]*high/size[1]
    return [int(width),int(high)]