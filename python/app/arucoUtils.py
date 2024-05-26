# from __future__ import division
import numpy as np
from numpy import linalg as LA
import cv2
import cv2.aruco as aruco
import transforms3d as tf3d
import time
from scipy.interpolate import griddata

from scipy.optimize import minimize, leastsq, least_squares
from scipy.spatial import distance
from matplotlib.path import Path
import math

_TF_end_face = np.loadtxt("python/dodecapen_assets/end_face_transforms.txt").reshape((-1, 3, 4))

frame_gray = np.zeros((100, 100, 3), np.uint8)
frame_gray_draw = np.copy(frame_gray)


def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0*v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis,:] + t_array[:,np.newaxis]*(v1 - v0)[np.newaxis,:]
        result = result/np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])


def patch_norm_and_grad(frame,frame_grad_u,frame_grad_v,corners_pix,bounding_box):
	''' patch making reference from
	    https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
 	'''
	a = bounding_box[0] 
	b = bounding_box[1] 
	start_pnt = [b[0],a[0]]
	 

	x, y = np.meshgrid(a, b) # make a canvas with coordinates
	x, y = x.flatten(), y.flatten()
	points = np.vstack((x,y)).T 

	p = Path(corners_pix) # make a polygon in pixel space
	grid = p.contains_points(points)  # make grid
	mask = grid.reshape(len(a),len(b)) 
	local_frame = frame[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)]
	local_frame_cropped = np.ones(local_frame.shape)

	np.copyto(local_frame_cropped,local_frame,where=mask)

	local_frame_cropped_norm = cv2.normalize(local_frame_cropped,None,alpha = 0, 
			beta=255,norm_type=cv2.NORM_MINMAX)

	local_frame_grad_v,local_frame_grad_u = np.gradient(local_frame_cropped_norm)

	# local_frame_grad_int8 = np.asarray(local_frame_grad,dtype=np.uint8)

	# np.copyto(frame_gray_draw[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)],
	# 	local_frame_grad_int8,where=mask)

	np.copyto(frame_grad_u[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)],
		local_frame_grad_u,where=mask)
 
	np.copyto(frame_grad_v[start_pnt[0]:start_pnt[0]+len(a), start_pnt[1]:start_pnt[1]+len(b)],
		local_frame_grad_v,where=mask)


	# cv2.imshow("frame",frame)
	# cv2.imshow("mask",np.array(mask*1*255,dtype=np.uint8))
	return frame_grad_v,frame_grad_u


def find_tfmat_avg(T_cent_accepted):
	Tf_cam_ball = np.eye(4)

	#### using slerp interpolation for averaging rotations
	sum_tra = np.zeros(3,)
	quat_av = tf3d.quaternions.mat2quat(T_cent_accepted[0][0:3,0:3])
	for itr in range(T_cent_accepted.shape[0]-1):
		quat2 = tf3d.quaternions.mat2quat(T_cent_accepted[itr+1][0:3,0:3])
		quat_av = slerp(quat2,quat_av,[0.5])
		quat_av =quat_av.reshape(4,)

	Tf_cam_ball[0:3,0:3]=tf3d.quaternions.quat2mat(quat_av)


	for itr in range(T_cent_accepted.shape[0]):
		sum_tra = sum_tra + T_cent_accepted[itr][0:3,3]

	sum_tra = sum_tra/T_cent_accepted.shape[0]
	Tf_cam_ball[0:3,3] = sum_tra
	
	return 	Tf_cam_ball
#######################################################


def RodriguesToTransf(x):
	'''
	Function to get a SE(3) transformation matrix from 6 Rodrigues parameters. NEEDS CV2.RODRIGUES()
	input: X -> (6,) (rvec,tvec)
	Output: Transf -> SE(3) rotation matrix
	'''
	x = np.array(x)
	x = x.reshape(6,)
	rot,_ = cv2.Rodrigues(x[0:3])
	trans =  np.reshape(x[3:6],(3,1))
	Transf = np.concatenate((rot,trans),axis = 1)
	Transf = np.concatenate((Transf,np.array([[0,0,0,1]])),axis = 0)
	return Transf


def LM_APE_Dodecapen(X, stacked_corners_px_sp, ids, params, flag=False):
	'''
	Function to get the objective function for APE step of the algorithm
	TODO: Have to put it in its respective class as a method (kind attn: Howard)
	Inputs: 
	X: (6,) array of pose parameters [rod_1, rod_2,rod_3,x,y,z]
	stacked_corners_px_sp = Output from Aruco marker detection. ALL the corners of the markers seen stacked in order 
	ids: int array of ids seen -- ids of faces seen
	Output: V = [4*M x 1] numpy array of difference between pixel distances
	'''
	# print(ids)
	corners_in_cart_sp = np.zeros((ids.shape[0],4,3))
	Tf_cam_ball = RodriguesToTransf(X)
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		corners_in_cart_sp[ii,:,:] = Tf_cam_ball.dot(corners_3d(Tf_cent_face, params.marker_size_in_mm)).T[:,0:3]
	
	corners_in_cart_sp = corners_in_cart_sp.reshape(ids.shape[0]*4,3)
	projected_in_pix_sp,_ = cv2.projectPoints(corners_in_cart_sp,np.zeros((3,1)),np.zeros((3,1)),
											  params.mtx,params.dist)

	projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)
	n,_=np.shape(stacked_corners_px_sp)
	V = LA.norm(stacked_corners_px_sp-projected_in_pix_sp, axis=1)
	
	if flag is False:
		return V


def tf_mat_dodeca_pen(face_id):
	'''
	Function that looks at the dodecahedron geometry to get the rotation matrix and translation
	TODO: when in class have to pass the dodecahedron geometry to this as a variable
	Inputs: face_id: the face for which the transformation matrices is quer=ries (int)
	Outputs: T_mat_cent_face = transformation matrix from center of the dodecahedron to a face
	T_mat_face_cent = transformation matrix from face (with given face id) to the dodecahedron center

	'''
	T_cent_face_curr = _TF_end_face[face_id - 1, :, 3:]
	_R_cent_face_curr = _TF_end_face[face_id - 1, :, :3]
	# print(T_cent_face_curr.shape, _R_cent_face_curr.shape)
	T_mat_cent_face = np.vstack((np.hstack((_R_cent_face_curr, T_cent_face_curr)), np.array([0, 0, 0, 1])))
	T_mat_face_cent = np.vstack(
		(np.hstack((_R_cent_face_curr.T, -_R_cent_face_curr.T.dot(T_cent_face_curr))), np.array([0, 0, 0, 1])))
	return T_mat_cent_face, T_mat_face_cent


def corners_3d(tf_mat,m_s):
	'''
	Function to give coordinates of the marker corners and transform them using a given transformation matrix
	Inputs:
	tf_mat = transformation matrix between face frames and camera frames
	m_s = marker size-- edge length in mm
	Outputs:
	corn_pgn_f = corners in camara frame
	'''
	corn_1 = np.array([-m_s/2.0,  m_s/2.0, 0, 1])
	corn_2 = np.array([ m_s/2.0,  m_s/2.0, 0, 1])
	corn_3 = np.array([ m_s/2.0, -m_s/2.0, 0, 1])
	corn_4 = np.array([-m_s/2.0, -m_s/2.0, 0, 1])
	corn_mf = np.vstack((corn_1,corn_2,corn_3,corn_4))
	corn_pgn_f = tf_mat.dot(corn_mf.T)
	return corn_pgn_f


def remove_bad_aruco_centers(center_transforms, params):

	"""
	takes in the tranforms for the aruco centers
	returns the transforms, centers coordinates, and indices for centers which
	aren't too far from the others
	Input: center_transforms  = N transformation matrices stacked as [N,4,4] numpy arrays
	Output = center_transforms[good_indices, :, :] -> accepted center transforms,
	centers_R3[good_indices, :] = center estimates from accepted center transforms,
	good_indices = accepted ids
	"""
	max_dist = 10 # pixels

	centers_R3 = center_transforms[:, 0:3, 3]
	projected_in_pix_sp,_ = cv2.projectPoints(centers_R3,np.zeros((3,1)),np.zeros((3,1)), params.mtx, params.dist) 
	
	projected_in_pix_sp = projected_in_pix_sp.reshape(projected_in_pix_sp.shape[0],2)


	distances = distance.cdist(centers_R3, centers_R3)
	distances_2 = distance.cdist(projected_in_pix_sp, projected_in_pix_sp)
	# print distances_2,"distances_2"
	# print distances,"distances"
	good_pairs = (distances_2 > 0) * (distances_2 < max_dist)

	good_indices = np.where(np.sum(good_pairs, axis=0) > 0)[0].flatten()

	if good_indices.shape[0] == 0 :
		# print('good_indices is none, resetting')
		good_indices = np.array([0, 1]) 

	return center_transforms[good_indices, :, :], centers_R3[good_indices, :], good_indices


def local_frame_grads (frame_gray, corners, ids,params): #### by arkadeep 
	''' Takes in the frame, the corners of the markers the camera sees and ids of the markers seen. 
	Returns the frame gradients
	Input: frame_gray --> grayscale frame
	corners: stacked as corners[num_markers,:,:] 
	ids: ids seen 
	Output
	frame_grad_u and frame_grad_v: matrices of sizes as frame gray. the areas near the markers will 
	have gradients of the frame_gray frame in the same locations and rest is 0
	'''
	 
	frame_grad_u = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
	frame_grad_u_temp = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
	frame_grad_v = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
	frame_grad_v_temp = np.zeros((frame_gray.shape[0],frame_gray.shape[1]))
	# debug_temp_img = np.zeros((frame_gray.shape[0],frame_gray.shape[1]),dtype= np.uint8)

	for i in range(len(ids)):
		expanded_corners_small = get_marker_borders(corners[i,:,:],params )
		v_low_small = int(np.min(expanded_corners_small[:,1]))
		v_high_small = int(np.max(expanded_corners_small[:,1])) 
		u_low_small = int(np.min(expanded_corners_small[:,0]))
		u_high_small = int(np.max(expanded_corners_small[:,0]))
		local_int_det = np.copy(frame_gray[v_low_small:v_high_small,u_low_small:u_high_small]) 
		max_int_small_sect = np.max(local_int_det)


		expanded_corners = get_marker_borders(corners[i,:,:],params)
		v_low = int(np.min(expanded_corners[:,1]))
		v_high = int(np.max(expanded_corners[:,1])) 
		u_low = int(np.min(expanded_corners[:,0]))
		u_high = int(np.max(expanded_corners[:,0]))

		frame_local = np.copy(frame_gray[v_low:v_high,u_low:u_high]) # not sure if v and u are correct order
 

		if abs(u_high - u_low) > abs(v_high- v_low): 
			sq_dim = u_high - u_low
		else:
			sq_dim = v_high - v_low

 
		# frame_local = cv2.normalize(frame_local,None,alpha = 0, 
		# 	beta=255.0*max_int_large_sect/max_int_small_sect,norm_type=cv2.NORM_MINMAX)
		# A,B = np.gradient(frame_gray )   
		# frame_grad_v  = np.copy(A) 
		# frame_grad_u  = np.copy(B)
		# cv2.imshow("full frame gradient",frame_grad_v.astype(np.uint8))		

		# A,B = np.gradient(frame_local)
		# frame_grad_v[v_low:v_high,u_low:u_high] = np.copy(A) 
		# frame_grad_u[v_low:v_high,u_low:u_high] = np.copy(B)
		# cv2.imshow("local frame gradient",frame_grad_v.astype(np.uint8))		

		frame_grad_v, frame_grad_u = patch_norm_and_grad(frame_gray,frame_grad_u_temp,frame_grad_v_temp,
			expanded_corners_small,[np.arange(u_low,u_low+sq_dim),np.arange(v_low,v_low+sq_dim)])
		# cv2.imshow("patched and normalised gradient",frame_grad_v.astype(np.uint8))		
		
		# cv2.waitKey(0)
 

	return  frame_grad_v, frame_grad_u


def marker_edges(ids, data,params):
	''' Function to give the edge points in the image and their intensities.
	to be called only once in the entire program to gather reference data for DPR
	output: 
	b_edge = [:,:] of size (ids x params.dwnsmpl_by,2) points on the marker in R3 where the intensities change form [x,y,0] stacked as 
	to be directly used in cv2.projectpoints
	edge_intensities = [:] ordered expected intensity points for the edge points to be used on obj fun of DPR size (ids x params.dwnsmpl_by)
	'''
	pix_offset = int((params.padding_fac-1)/2 * 600)  ### aruco images are 600x600
	b_edge = []
	edge_intensities_expected = []

	for aruco_id in ids:
		b = data.edge_pts_in_img_sp[aruco_id[0]]
		n = b.shape[0]
		b[:,2] = 0.
		b[:,3] = 1
		b_shaped = b[0::params.dwnsmpl_by,0:4].astype(np.float32) 
		b_edge.append(b_shaped)
		img_pnts_curr = data.img_pnts[aruco_id[0]][0::params.dwnsmpl_by,:]
		edge_intensities = data.aruco_images_int16[aruco_id[0]][img_pnts_curr [:,1]+pix_offset,img_pnts_curr [:,0]+pix_offset]# TODO can we have it in terms of dil_fac
		edge_intensities_expected.append(edge_intensities)
	# ---------------------------------------------------------------

 	########## this part varifies that correct points are sampled from the edges of the aruco images. 
	# for j in range(len(ids)):
	# 	img_pnts_curr =data.img_pnts[ids[j][0]][0::params.dwnsmpl_by,:]
	# 	n_int = img_pnts_curr.shape[0]
	# 	print (img_pnts_curr.shape,"img_pnts_curr.shape")
	# 	print (data.img_pnts[ids[j][0]].shape,"img_pnts.shape")
	# 	for i in range(n_int):
	# 		center_2 = tuple(np.ndarray.astype(np.array([img_pnts_curr[i,0]+pix_offset,img_pnts_curr[i,1]+pix_offset]),int))
	# 		cv2.circle( data.aruco_images[ids[j][0]], center_2 , 3 , 127, -1)
	# 	cv2.imshow("data.aruco_images[ids_{}]".format(ids[j][0]),data.aruco_images[ids[j][0]])
	# 	cv2.waitKey(0)	

	# print (edge_intensities_expected[0]," edge_intensities_expected[{}]".format(ids[0][0]))
	# print (ids[0])
	# print (b_edge[0].shape,"b_edge[0].shape")
	# print (edge_intensities_expected[0].shape,"edge_intensities_expected[0].shape")
	# print ("")	

		# y = data.aruco_images_int16[aruco_id][data.img_pnts[aruco_id][:,1]+pix_offset,data.img_pnts[aruco_id][:,0]+pix_offset]

	# return np.asarray(b_edge) , np.asarray(edge_intensities_expected)
	return b_edge, edge_intensities_expected
	# pass


def get_marker_borders (corners,params):
	''' Dilates a given marker from the corner pxl locations in an image by the dilate factor. 
	Returns: stack of expanded corners in the pixel space'''

	cent = np.array([np.mean(corners[:,0]), np.mean(corners[:,1])])
	
	vert_1 = (corners[0,:] - cent)* params.dilate_fac
	vert_2 = (corners[1,:] - cent)* params.dilate_fac
	vert_3 = (corners[2,:] - cent)* params.dilate_fac
	vert_4 = (corners[3,:] - cent)* params.dilate_fac
	
	expanded_corners = np.vstack((vert_1+cent,vert_2+cent,vert_3+cent,vert_4+cent))
	 
	return expanded_corners
 

def LM_DPR(X, frame_gray, ids, corners, b_edge, edge_intensities_expected_all, data, params):
	''' Objective function for the DPR step. Takes in pose as the first arg [mandatory!!] and,
	returns the value of the obj fun and jacobial of the obj fun'''    
	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		borders_in_cart_sp.append((Tf_cam_ball.dot(Tf_cent_face).dot(b_edge[ii].T)).T)
		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)

	proj_points, _ = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], np.zeros((3,1)), np.zeros((3,1)), params.mtx, params.dist)
	proj_points_int = np.ndarray.astype(proj_points,int)

	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)
	n_int = proj_points_int.shape[0]
	temp = proj_points.shape[0]
	proj_points = proj_points.reshape(temp,2)
	# testing_points = 10 
 	# LM_DPR_DRAW(X, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected_all,data, params, 127)
 	# -profile do not put drawing stuff here save 13% time
	f_p = frame_gray[proj_points_int[:,1],proj_points_int[:,0]]  # TODO i dont think framegray int16 is needed ? Also 0,1 order changed
	err = (edge_intensities_expected_stacked/1.0 - f_p.reshape(f_p.shape[0],1)/1.0) # this is the error in the intensities
	return  err.reshape(err.shape[0],)


def LM_DPR_DRAW(X, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected_all, data, params, col_gr = 127, rad=1):
	''' Objective function for the DPR step. Takes in pose as the first arg [mandatory!!] and,
	returns the value of the obj fun and jacobial of the obj fun'''    
	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		borders_in_cart_sp.append((Tf_cam_ball.dot(Tf_cent_face).dot(b_edge[ii].T)).T)
		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)

	proj_points, _ = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], np.zeros((3,1)), np.zeros((3,1)), params.mtx, params.dist)
	proj_points_int = np.ndarray.astype(proj_points,int)

	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)
	n_int = proj_points_int.shape[0]
	temp = proj_points.shape[0]
	proj_points = proj_points.reshape(temp,2)
	
	testing_points = temp 
	for i in range(n_int):
		center = tuple(np.ndarray.astype(proj_points_int[i,:],int))
		cv2.circle( frame_gray_draw, center , rad , col_gr, -1)
	

	# f_p = frame_gray[proj_points_int[:,1],proj_points_int[:,0]]  # TODO i dont think framegray int16 is needed ? Also 0,1 order changed
	# err = (edge_intensities_expected_stacked - f_p.reshape(f_p.shape[0],1))/1.0 # this is the error in the intensities
	return  frame_gray_draw


def LM_DPR_Jacobian(X, frame_gray, ids, corners, b_edge, edge_intensities_expected_all, data, params):

	'''Function to calculate the Jacobian of the objective function'''
 
	Tf_cam_ball = RodriguesToTransf(X)
	borders_in_cart_sp = []
	edge_intensities_expected = []
	for ii in range(ids.shape[0]):
		Tf_cent_face,Tf_face_cent = tf_mat_dodeca_pen(int(ids[ii]))
		# borders_in_cart_sp.append((Tf_cam_ball.dot(b_edge[ii].T)).T)  ### by arkadeep
		borders_in_cart_sp.append(( Tf_cent_face.dot(b_edge[ii].T)).T)  ### added by tejas

		edge_intensities_expected.append(edge_intensities_expected_all[ii].reshape(edge_intensities_expected_all[ii].shape[0],1))

	stacked_borders_in_cart_sp = np.vstack(borders_in_cart_sp)
	edge_intensities_expected_stacked = np.vstack(edge_intensities_expected)
	proj_points , duvec_by_dp_all = cv2.projectPoints( stacked_borders_in_cart_sp [:,0:3], X[0:3],X[3:6], params.mtx, params.dist)

	proj_points_int = np.ndarray.astype(proj_points,int) # -profile this is taking some time (5.5% of total time)

	proj_points = proj_points.reshape(proj_points.shape[0],2)
	proj_points_int = proj_points_int.reshape(proj_points_int.shape[0],2)
	
	du_by_dp = griddata(proj_points,duvec_by_dp_all[0::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
	dv_by_dp = griddata(proj_points,duvec_by_dp_all[1::2,0:6],(proj_points_int[:,0],proj_points_int[:,1]), method = 'nearest')
	
	# this is same as above - no it's not
	# du_by_dp = duvec_by_dp_all[0::2,0:6]
	# dv_by_dp = duvec_by_dp_all[1::2,0:6] 
	# print np.linalg.norm(du_by_dp- du_by_dp_grid),"norm"
	# print ""

	dI_by_dv ,dI_by_du = local_frame_grads (frame_gray.astype('int16'), np.vstack(corners), ids,params) ##TODO local frame gradients not working pl check ## arkadeep
 	 

 	# LM_DPR_DRAW(X, frame_gray_draw, ids, corners, b_edge, edge_intensities_expected_all,data, mtx, dist, 127,1)


	n_int = proj_points_int.shape[0]
	dI_by_dp = np.zeros((n_int,6))
 

	for i in range(n_int):
		ui,vi = proj_points_int[i,0], proj_points_int[i,1]
		# dI_by_dp[i,:] = dI_by_du [ui,vi] * du_by_dp[i] + dI_by_dv [ui,vi] * dv_by_dp[i] #TODO confirn [u,v] order in eqn  # by arkadeep
		dI_by_dp[i,:] = dI_by_du [vi,ui] * du_by_dp[i] + dI_by_dv [vi,ui] * dv_by_dp[i] #TODO confirn [u,v] order in eqn  #edited  by tejas
 
	return -dI_by_dp  # the neg sign is required

######

# def get_pnt_grey_image():
# 	rospy.init_node('RosCam', anonymous=True)
# 	ic = RosCam("/camera/image_color")
# 	return ic

class ddc_parameters():
	def __init__(self, depth_cam_matrix, depth_cam_dist):
		self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
		# self.aruco_params = aruco.DetectorParameters_create()
		self.aruco_params = aruco.DetectorParameters()
		self.aruco_params.cornerRefinementMethod = 1
		self.aruco_params.cornerRefinementMinAccuracy = 0.05
		self.marker_size_in_mm = 10.8#12.96
		self.dilate_fac = 1.2 # dilate the square around the marker
		self.padding_fac = 1.2 # padding around the aruco source image DO NOT CHANGE THIS PARAMETER
		# with np.load('PTGREY.npz') as X: # Camera projection matrix and distortion coefficient, TODO: get from holoens
		# 	cam_mtx, cam_dist = [X[i] for i in ('mtx','dist')]
		self.mtx = depth_cam_matrix
		self.dist = depth_cam_dist
		self.dwnsmpl_by = 100           ## changed the number of points_for_DPR
		self.markers_possible = np.array([1,2,3,4,5,6,7,8,10,11,12])
		self.markers_impossible = np.array([0,9,13,17,37,16,34,45,38,24,47,32,40])

		self.tip_loc_cent = np.array([9.21838242, 139.40608895, -87.80389854, 1]).reshape(4, 1)


class ddc_txt_data():
	def __init__(self, txt_file_path):
		self.x = 0
		self.edge_pts_in_img_sp = [0] * 13
		self.aruco_images = [0] * 13
		self.aruco_images_int16 = [0] * 13
		self.img_pnts = [0] * 13
		for i in range(1, 13):
			self.edge_pts_in_img_sp[i] = np.loadtxt(f"{txt_file_path}/thick_edge_coord_R3/id_{i}.txt", delimiter=',', dtype='float32') * 10.8/17.78
			self.aruco_images[i] = cv2.imread(f"{txt_file_path}/aruco_images_mip_maps/res_75_{i}.jpg", 0)
			self.img_pnts[i] = np.loadtxt(f"{txt_file_path}/thick_edge_coord_pixels/id_{i}.txt", delimiter=',', dtype='int16')
			self.aruco_images_int16[i] = np.int16(self.aruco_images[i])


class OneEuroFilter:
	def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
		"""Initialize the one euro filter."""
		# The parameters.
		self.min_cutoff = float(min_cutoff)
		self.beta = float(beta)
		self.d_cutoff = float(d_cutoff)

		# Previous values.
		self.x_prev = x0
		self.dx_prev = float(dx0)
		self.t_prev = t0

	def smoothing_factor(self, t_e, cutoff):
		r = 2 * math.pi * cutoff * t_e
		return r / (r + 1)

	def exponential_smoothing(self, a, x, x_prev):
		return a * x + (1 - a) * x_prev

	def filter_signal(self, t, x):
		"""Compute the filtered signal."""
		t_e = t - self.t_prev

		# The filtered derivative of the signal.
		a_d = self.smoothing_factor(t_e, self.d_cutoff)
		dx = (x - self.x_prev) / t_e
		dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

		# The filtered signal.
		cutoff = self.min_cutoff + self.beta * abs(dx_hat)
		a = self.smoothing_factor(t_e, cutoff)
		x_hat = self.exponential_smoothing(a, x, self.x_prev)

		# Memorize the previous values.
		self.x_prev = x_hat
		self.dx_prev = dx_hat
		self.t_prev = t

		return x_hat



