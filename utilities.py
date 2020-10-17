import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def square_pad(image):
    sq_size = max(image.shape)
    return cv.copyMakeBorder(image, 0, sq_size - image.shape[0], 0, sq_size - image.shape[1], cv.BORDER_CONSTANT)

def re_center_scale(peaks):
    avai_peaks = peaks[peaks.sum(axis=1) > 0]
    box = [avai_peaks[:,0].min(),avai_peaks[:,1].min(),avai_peaks[:,0].max(),avai_peaks[:,1].max()]
    center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    new_peaks = peaks.copy()
    new_peaks[new_peaks.sum(axis=1) > 0] -= np.array(center)
    ratio = min([-0.4/(avai_peaks[:,0].min() - center[0]),
                 -0.4/(avai_peaks[:,1].min() - center[1]),
                 0.4/(avai_peaks[:,0].max() - center[0]),
                 0.4/(avai_peaks[:,1].max() - center[1])])
    re_peaks = new_peaks.copy()
    re_peaks[re_peaks!=-1] *= ratio
    re_peaks[re_peaks!=-1] += 0.5
    return re_peaks

def plot_pose(peaks,figsize=(5,5),linecolor='blue',pointcolor='red'):
    fig = plt.figure(figsize=figsize)
#     pyplot.rcParams['xtick.bottom'] = pyplot.rcParams['xtick.labelbottom'] = False
#     pyplot.rcParams['xtick.top'] = pyplot.rcParams['xtick.labeltop'] = True

        
    def draw_line(i,j):
        if np.all(peaks[i]>0) and np.all(peaks[j]>0):
            plt.plot([peaks[i][0],peaks[j][0]],[peaks[i][1],peaks[j][1]], color=linecolor)
        
    #draw lines
    draw_line(0,1)
    draw_line(1,2)
    draw_line(1,5)
    draw_line(2,3)
    draw_line(3,4)
    draw_line(1,5)
    draw_line(5,6)
    draw_line(6,7)
    draw_line(1,8)
    draw_line(1,11)
    draw_line(8,9)
    draw_line(9,10)
    draw_line(11,12)
    draw_line(12,13)
    
    #draw peaks
    for peak in peaks:
        plt.plot([peak[0]],[peak[1]],marker='o',markersize=10,color=pointcolor)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()
    
##plot a sequence of multiple frames
def plot_sequence(sequence,imsize=(2,2),linecolor='blue',pointcolor='red'):
    n_frames = len(sequence)
    figsize = (imsize[0]*1.2*n_frames, imsize[1])
    plt.rcParams["figure.figsize"] = figsize
    fig = plt.figure()

    f, axarr = plt.subplots(1, n_frames, sharey=True)
    axarr[0].invert_yaxis()
    
    def draw_line(ax,i,j):
        if np.all(peaks[i]>0) and np.all(peaks[j]>0):
            ax.plot([peaks[i][0],peaks[j][0]],[peaks[i][1],peaks[j][1]], color=linecolor);
            
    for i in range(n_frames):
        axarr[i].set_box_aspect(1)
        
        peaks = sequence[i]
        
        draw_line(axarr[i],0,1)
        draw_line(axarr[i],1,2)
        draw_line(axarr[i],1,5)
        draw_line(axarr[i],2,3)
        draw_line(axarr[i],3,4)
        draw_line(axarr[i],1,5)
        draw_line(axarr[i],5,6)
        draw_line(axarr[i],6,7)
        draw_line(axarr[i],1,8)
        draw_line(axarr[i],1,11)
        draw_line(axarr[i],8,9)
        draw_line(axarr[i],9,10)
        draw_line(axarr[i],11,12)
        draw_line(axarr[i],12,13)

        #draw peaks
        for peak in peaks:
            axarr[i].plot([peak[0]],[peak[1]],marker='o',markersize=5,color=pointcolor);
            axarr[i].set_xlim([0, 1])
            axarr[i].set_ylim([0, 1])
            axarr[i].invert_yaxis()
    plt.show()
    
def rotate_pose(pose,theta_x,theta_y):
    pose_copy = pose.copy() - pose.mean(axis=0)
    for p in pose_copy:
        p[0] = p[0]*np.cos(theta_y) + p[1]*np.sin(theta_x)*np.sin(theta_y)
        p[1] = p[1]*np.cos(theta_x)
    return pose_copy + pose.mean(axis=0)

def rotate_join(pose, peak, joint, theta):
    cjoints = pose[joint] - pose[peak]
    rtpose = pose.copy()
    rtpose[joint,0] = cjoints[:,0]*np.cos(theta) - cjoints[:,1]*np.sin(theta)
    rtpose[joint,1] = cjoints[:,0]*np.sin(theta) + cjoints[:,1]*np.cos(theta)
    rtpose[joint] += pose[peak]
    return rtpose

def rot_head(pose, r_sd):
    return rotate_join(pose, 1, [0], np.random.normal(0, r_sd))
       
def rot_elbow_left(pose, r_sd):
    return rotate_join(pose, 3, [4], np.random.normal(0, r_sd))
    
def rot_elbow_right(pose, r_sd):
    return rotate_join(pose, 6, [7], np.random.normal(0, r_sd))

def rot_arm_left(pose, r_sd):
    return rotate_join(pose, 2, [3, 4], np.random.normal(0, r_sd))
    
def rot_arm_right(pose, r_sd):
    return rotate_join(pose, 5, [6, 7], np.random.normal(0, r_sd))    

def rot_knee_left(pose, r_sd):
    return rotate_join(pose, 9, [10], np.random.normal(0, r_sd))
    
def rot_knee_right(pose, r_sd):
    return rotate_join(pose, 12, [13], np.random.normal(0, r_sd))

def rot_leg_left(pose, r_sd):
    return rotate_join(pose, 8, [9, 10], np.random.normal(0, r_sd))
    
def rot_leg_right(pose, r_sd):
    return rotate_join(pose, 11, [12, 13], np.random.normal(0, r_sd))

all_rots = np.array([rot_head, rot_elbow_left, rot_elbow_right, rot_arm_left, rot_arm_right, 
                     rot_knee_left, rot_knee_right, rot_leg_left, rot_leg_right])

def randomize_pose(pose, r_sd, p_sd):
    rand_rots = np.random.choice(all_rots, np.random.randint(0,10))
    rpose = pose.copy()
    
    for rotate in rand_rots:
        rpose = rotate(pose, r_sd)
    
    x = np.random.normal(0, p_sd, 2)
    t = x*(x>0) + (2*np.pi + x)*(x<0)

    return rotate_pose(rpose, t[0], t[1])

def randomize_set(poses, r_sd, p_sd):
    return np.array([randomize_pose(p, r_sd, p_sd) for p in poses])

def overlay_pose(peaks1, peaks2, figsize=(5,5),linecolor=['blue','orange'],pointcolor=['red','green']):
    fig = plt.figure(figsize=figsize)

    def draw_pose(peaks,p):
        def draw_line(i,j):
            if np.all(peaks[i]!=0) and np.all(peaks[j]!=0):
                plt.plot([peaks[i][0],peaks[j][0]],[peaks[i][1],peaks[j][1]], color=linecolor[p])

        #draw lines
        draw_line(0,1)
        draw_line(1,2)
        draw_line(1,5)
        draw_line(2,3)
        draw_line(3,4)
        draw_line(1,5)
        draw_line(5,6)
        draw_line(6,7)
        draw_line(1,8)
        draw_line(1,11)
        draw_line(8,9)
        draw_line(9,10)
        draw_line(11,12)
        draw_line(12,13)

        #draw peaks
        for peak in peaks:
            plt.plot([peak[0]],[peak[1]],marker='o',markersize=10,color=pointcolor[p])

    draw_pose(peaks1, 0)
    draw_pose(peaks2, 1)
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()