import numpy as np
import math
import random

class Rain:
    def __init__(self,M,B,psi):
        self.M=M
        self.B=B
        self.psi=psi/180.0*math.pi
        self.n_air=1.0
        self.n_water=1.33
        self.gamma=math.asin(self.n_air/self.n_water)
        self.normal=np.array([0,-1.0*math.cos(self.psi),math.sin(self.psi)])
        val=self.normal.dot(self.normal)
        self.o_g=self.normal[-1]*self.M/val*self.normal
        
    def get_intrinsic(self,cx,cy,fx,fy):
        self.intrinsic=np.eye(3)
        self.intrinsic[0,0]=fx
        self.intrinsic[1,1]=fy
        self.intrinsic[0,2]=cx
        self.intrinsic[1,2]=cy
        self.cxy=np.array([cx,cy])
        self.fxy=np.array([fx,fy])
        
    def get_kernel(self,kernel_sz):
        kernel=np.zeros((kernel_sz,kernel_sz))
        radius=kernel_sz/2.0
        idx_kernel=np.indices(kernel.shape)
        yx=np.hstack((idx_kernel[0,:,:].reshape((-1,1)),idx_kernel[1,:,:].reshape((-1,1))))

        val=np.linalg.norm(yx-radius,axis=1)
        idx=np.where(val<radius)
        yx_id=yx[idx]
        # print(idx)
        # print(yx.shape)
        # print(yx[idx].shape)
  
        kernel[yx_id[:,0],yx_id[:,1]]=1
        # print(kernel,yx.shape)
        # exit()
        count=idx[0].shape[0]
        kernel=kernel/count
        return kernel
        
    #@staticmethod
    def to_glass(self,xy):
        tan_psi=math.tan(self.psi)
        
        w=self.M*tan_psi/(tan_psi-(xy[:,1]- self.intrinsic[1,2])/self.intrinsic[1,1])
        uv_tmp=(xy-self.cxy)/self.fxy
        w=np.expand_dims(w,axis=1)
        return np.hstack( (uv_tmp*w,w) )

    #@staticmethod
    def w_in_plane(self,uv):
        return self.M-uv.dot(self.normal[0:2])/self.normal[-1]
    
    #@staticmethod
    def get_sphere_raindrop(self,wh):
   
        left_upper=self.to_glass(np.array([[0,0]]))
        left_bottom=self.to_glass(np.array([[0,wh[1]]]))
        #right_upper=self.to_glass(wh[0],0)
        right_bottom=self.to_glass(np.array([[wh[0],wh[1]]]))
        random.seed()
        random_rain=random.randint(100,200)
        loc=[]
        tau=[]
        uv=[]
        for i in range(random_rain):
            random.seed()
            random_loc=random.random()
            u=left_bottom[0,0]+(right_bottom[0,0]-left_bottom[0,0])*random_loc
            random.seed()
            random_loc=random.random()
            v=left_upper[0,1]+(right_bottom[0,1]-left_upper[0,1])*random_loc
            uv.append([u,v])
            
            random.seed()
            tau.append(random.randint(30,45)/180.0*math.pi)
            random.seed()
            random_loc=random.random()
            loc.append(random_loc) 
        
        random_loc=np.array(loc)
        random_tau=np.array(tau)
        uv=np.array(uv)
        glass_r=0.8+0.6*random_loc
        r_sphere=glass_r/np.sin(random_tau)

        w=self.w_in_plane(uv).reshape((-1,1))
        g_c=np.hstack((uv,w))
        c=g_c-(r_sphere*np.cos(random_tau)).reshape((-1,1))*(self.normal.reshape((1,-1)))
        
        return g_c,glass_r,c,r_sphere
    

    
    def to_sphere_section_env(self,xy,g_centers,g_radius,centers,radius):

        p=self.to_glass(xy)
        idx=[]
        p_g=[]
        xy_pos=[]
        for i,p_v in enumerate(p):
            ask=p_v.reshape((1,-1))-g_centers
            f=np.linalg.norm(ask,axis=1)
            #print(f.shape,ask.shape)
            valid_idx=np.where(f<=g_radius)
            if valid_idx[0].shape[0]==0:
                continue
            idx.append(valid_idx[0][0])
            p_g.append(p_v)
            xy_pos.append(xy[i,:])
        if len(idx)==0:
            return None,None,None
          
        xy_pos=np.array(xy_pos)
        p_g=np.array(p_g)

        alpha=np.arccos(p_g.dot(self.normal)/np.linalg.norm(p_g,axis=1) )
        beta=np.arcsin(self.n_air*np.sin(alpha)/self.n_water)
        po=p_g-self.o_g
        po_norm=np.linalg.norm(po,axis=1)
        po=(po.T/po_norm).T
        i_1=self.normal+po*(np.tan(beta).reshape((-1,1)))
        i_1_norm=np.linalg.norm(i_1,axis=1)
        i_1=(i_1.T/i_1_norm).T
        calc_centers=centers[idx]
        calc_radius=radius[idx]

        oc=p_g-calc_centers
        tmp=np.sum(i_1*oc,axis=1)
        

        d=-tmp+np.sqrt(np.power(tmp,2)-np.sum(oc*oc,axis=1)+np.power(calc_radius,2) )
        calc_d=d.reshape((-1,1))
        p_w=p_g+i_1*calc_d

        normal_w=p_w-calc_centers
        normal_w_norm=np.linalg.norm(normal_w,axis=1)
        normal_w=(normal_w.T/normal_w_norm).T

        pwg=(p_w-p_g)
        pwg_norm=np.linalg.norm(pwg,axis=1)
        d=np.sum(pwg*normal_w,axis=1)/np.sum(normal_w*normal_w,axis=1)
        calc_d=d.reshape((-1,1))
        p_a=pwg-normal_w*calc_d
        p_a_norm=np.linalg.norm(p_a,axis=1)
        p_a=(p_a.T/p_a_norm).T
        
        eta=np.arccos(np.sum(normal_w*pwg,axis=1)/pwg_norm)
        valid_idx=np.where(eta<self.gamma)
        invalid_idx=np.where(eta>=self.gamma)


        
        xy_pos_invalid=xy_pos[invalid_idx]
        
        if valid_idx[0].shape[0]==0:
            return None,None,xy_pos_invalid
        
        xy_pos_valid=xy_pos[valid_idx]
        eta=eta[valid_idx]
        p_w=p_w[valid_idx]
        p_a=p_a[valid_idx]
        normal_w=normal_w[valid_idx]
        theta=np.arcsin(self.n_water*np.sin(eta)/self.n_air)

        i_2=normal_w+p_a*(np.tan(theta).reshape((-1,1)))

        calc_z=(self.B-p_w[:,-1])/i_2[:,-1]
        p_e=p_w+i_2*(calc_z.reshape((-1,1)))

        z=self.intrinsic@(p_e.T)/self.B
        z=np.round(z.T).astype(np.int32)
        return z,xy_pos_valid,xy_pos_invalid
    
    def render(self,image):
        rain_image=image.copy()
        mask=np.zeros(image.shape[0:2],np.uint8)
        
        self.imgh,self.imgw=image.shape[0:2]
        self.get_intrinsic(self.imgw/2,self.imgh/2,self.imgw*1.1,self.imgw*1.1)
        g_c,glass_r,c,r_sphere=self.get_sphere_raindrop([self.imgw,self.imgh])
        idx_img = np.indices((self.imgw,self.imgh))
        xy=np.hstack((idx_img[0,:,:].reshape((-1,1)),idx_img[1,:,:].reshape((-1,1))))
        z,xy_pos_valid,xy_pos_invalid=self.to_sphere_section_env(xy,g_c,glass_r,c,r_sphere)
        if xy_pos_invalid is None:
            return
        rain_image[xy_pos_invalid[:,1],xy_pos_invalid[:,0],:]=0
        
        if xy_pos_valid is None:
            return
        u=np.clip(z[:,0],0,self.imgw-1)
        v=np.clip(z[:,1],0,self.imgh-1)
        rain_image[xy_pos_valid[:,1],xy_pos_valid[:,0],:]=image[v,u,:]
        
        mask[xy_pos_invalid[:,1],xy_pos_invalid[:,0]]=255
        mask[xy_pos_valid[:,1],xy_pos_valid[:,0]]=255
        return rain_image,mask
    
    def blur(self,kernel,image,rain_image,mask):
        blur_image=image.copy()
        blured=cv2.filter2D(rain_image,-1,kernel)
        blured=(blured*1.2).astype(np.uint8)
        np.copyto(blur_image,blured,where=mask[:,:,None].astype(np.bool))
        
        return blur_image

        
        
        
import cv2
if __name__=='__main__':
    
    image=cv2.imread('./figure/demo.jpg')
    
    random.seed()
    M=random.randint(100,350/2)
    random.seed()
    B=random.randint(4000,8000)
    random.seed()
    psi=random.randint(30,45)
    random.seed()
    ksz=random.randint(5,8)    
    rain=Rain(M,B,psi)
    kernel=rain.get_kernel(ksz)
    print(M,B,psi,ksz)
    rain_image,mask=rain.render(image)
    blur_image=rain.blur(kernel,image,rain_image,mask)
    #cv2.imwrite('ss.jpg',rain_image)
    cv2.imwrite('./figure/demo_res.jpg',blur_image)
        
            
