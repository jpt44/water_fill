import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad


def integrand(a,b,c,x):
    return a*x**2+b*x+c

def quadratic_fit(data):
    """
    Input: indices of values not equal to 0

    Outputs (3): coefficients, Root mean square error,
    and R_squared
    """

    x=data[:,1].reshape(-1,1)
    y=data[:,0].reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression(n_jobs=2)
    reg = model.fit(x_poly, y)

    coef=reg.coef_

    return coef

def water_fill6(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    if indices.size!=0:
        topRow,leftColumn=np.min(indices,axis=0)
        bottomRow,rightColumn=np.max(indices,axis=0)
        image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]
    else:
        h, w = image.shape
        return 0.0, 0.0, 0.0, 0.0, h, w

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    tes=np.argwhere(image[:,w//2]!=0)
    if tes.size!=0:
        rw=np.min(tes[:,0])
        col=w//2
        partial_img = image[:rw + 1, :]
        tes = np.argwhere(image[:, w // 3] != 0)
        if tes.size!=0 and np.min(tes[:,0])>rw:
            rw=np.min(np.argwhere(image[:,w//3]!=0)[:,0])
            col=w//3
            partial_img = image[:rw + 1, :col+1]
        tes = np.argwhere(image[:, 2*w // 3] != 0)
        if tes.size!=0 and np.min(tes[:,0])>rw:
            rw = np.min(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
            col=2*w//3
            partial_img = image[:rw + 1, col:]
        indices = np.argwhere(partial_img != 0)  # row,column
        indices[:, 0] = rw + 1 - indices[:, 0]  # so image is outputted upright
        coef=quadratic_fit(indices)
        lb=np.min(indices[:,1]) #lower bound to integrate over
        ub=np.max(indices[:,1]) #upper bound to integrate over
        Iup = quad(integrand, a=lb, b=ub,
                   args=(coef[0,2], coef[0,1], coef[0,0]))
        y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
        y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
        if coef[0,2]<0:
            Iup=abs(Iup[0])
        else:
            rectHeight = max(y1,y2)
            Iup=rectHeight*abs(lb-ub)-abs(Iup[0])
    else:
        Iup=0

    #Calculate down jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    tes=np.argwhere(image[:, w // 2] != 0)
    if tes.size != 0:
        rw=np.min(tes[:,0])
        col=w//2
        partial_img=image[rw:,:]
        tes=np.argwhere(image[:,w//3]!=0)
        if tes.size!=0 and np.min(tes[:,0])<rw:
            rw=np.min(np.argwhere(image[:,w//3]!=0)[:,0])
            col=w//3
            partial_img = image[rw:,:col+1]
        tes = np.argwhere(image[:, 2*w // 3] != 0)
        if tes.size!=0 and np.min(tes[:,0])<rw:
            rw = np.min(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
            col=2*w//3
            partial_img = image[rw:, col:]
        indices = np.argwhere(partial_img != 0)  # row,column
        indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
        coef=quadratic_fit(indices)
        lb=np.min(indices[:,1]) #lower bound to integrate over
        ub=np.max(indices[:,1]) #upper bound to integrate over
        Idown = quad(integrand, a=lb, b=ub,
                   args=(coef[0,2], coef[0,1], coef[0,0]))
        y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
        y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
        if coef[0,2]<0:
            Idown=abs(Idown[0])
        else:
            rectHeight = max(y1,y2)
            Idown=rectHeight*abs(lb-ub)-abs(Idown[0])
    else:
        Idown=0

    #Calculate right jar capacity
    # Find column furthest right that !=0 in middle of image
    tes=np.argwhere(image[h//2,:]!=0)
    if tes.size != 0:
        col=np.max(tes[:,0])
        rw=h//2
        partial_img=image[:,col:]
        tes = np.argwhere(image[h // 3, :] != 0)
        if tes.size!=0 and np.max(tes[:,0])<col:
            col=np.max(tes[:,0])
            rw=h//3
            partial_img = image[:rw+1,col:]
        tes = np.argwhere(image[2*h // 3, :] != 0)
        if tes.size!=0 and np.max(tes[:,0])<col:
            col = np.max(tes[:, 0])
            rw=2*h//3
            partial_img = image[rw:, col:]
        indices = np.argwhere(partial_img != 0)  # row,column
        indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
        indices=np.flip(indices,axis=1) #swap row and column indices
        coef=quadratic_fit(indices)
        lb=np.min(indices[:,1]) #lower bound to integrate over
        ub=np.max(indices[:,1]) #upper bound to integrate over
        Iright = quad(integrand, a=lb, b=ub,
                   args=(coef[0,2], coef[0,1], coef[0,0]))
        y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
        y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
        if coef[0,2]<0:
            Iright=abs(Iright[0])
        else:
            rectHeight = max(y1,y2)
            Iright=rectHeight*abs(lb-ub)-abs(Iright[0])
    else:
        Iright=0

    #Calculate left jar capacity
    # Find column furthest right that !=0 in middle of image
    tes=np.argwhere(image[h//2,:]!=0)
    if tes.size!=0:
        col=np.min(tes[:,0])
        rw = h // 2
        partial_img=image[:,:col+1]
        tes = np.argwhere(image[h // 3, :] != 0)
        if tes.size!=0 and np.min(tes[:,0])>col:
            col=np.min(tes[:,0])
            rw=h//3
            partial_img = image[:rw+1,:col+1]
        tes = np.argwhere(image[2*h // 3, :] != 0)
        if tes.size!=0 and np.min(tes[:,0])>col:
            col = np.min(tes[:, 0])
            rw=2*h//3
            partial_img = image[rw:, :col+1]
        indices = np.argwhere(partial_img != 0)  # row,column
        indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
        indices=np.flip(indices,axis=1) #swap row and column indices
        coef=quadratic_fit(indices)
        lb=np.min(indices[:,1]) #lower bound to integrate over
        ub=np.max(indices[:,1]) #upper bound to integrate over
        Ileft = quad(integrand, a=lb, b=ub,
                   args=(coef[0,2], coef[0,1], coef[0,0]))
        y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
        y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
        if coef[0,2]<0:
            Ileft=abs(Ileft[0])
        else:
            rectHeight = max(y1,y2)
            Ileft=rectHeight*abs(lb-ub)-abs(Ileft[0])
    else:
        Ileft=0

    return Iup,Idown,Ileft,Iright,h,w