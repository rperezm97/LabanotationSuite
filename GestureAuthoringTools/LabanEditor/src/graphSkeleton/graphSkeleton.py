# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os, math, copy
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from human_body_prior.body_model.body_model import BodyModel
import torch
try:
    from tkinter import messagebox
except ImportError:
    # Python 2
    import tkMessageBox as messagebox

import settings

#------------------------------------------------------------------------------
# AMASS Joint Mapping to Kinect Body Format


class graph3D:
    fig = None
    ax = None
    jointFrames = []    # all the frames
    labanjointFrames = []
    all_laban = []
    timeS = []

    joints = []         # temporary joints to be drawn
   
    # a joint point, 'ts' stands for tracking status
    jType = np.dtype({'names':['x', 'y', 'z','ts'],'formats':[float,float,float,int]})

    # a body
    bType = np.dtype({'names':[ 'timeS',                # milliseconds
                                'filled',               # filled gap
                                'spineB', 'spineM',     # meter
                                'neck', 'head',
                                'shoulderL', 'elbowL', 'wristL', 'handL', # tracked=2, inferred=1, nottracked=0
                                'shoulderR', 'elbowR', 'wristR', 'handR',
                                'hipL', 'kneeL', 'ankleL', 'footL',
                                'hipR', 'kneeR', 'ankleR', 'footR',
                                'spineS', 'handTL', 'thumbL', 'handTR', 'thumbR'],
                    'formats':[ int,
                                bool,
                                jType, jType,
                                jType, jType,
                                jType, jType, jType, jType,
                                jType, jType, jType, jType,
                                jType, jType, jType, jType,
                                jType, jType, jType, jType,
                                jType, jType, jType, jType, jType]})
    sq_error=[]
    limb_lenghts=[]
    isInterpolatedKeyFrame = False
    annSelection = None
    

    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self, filePath=None):
        self.strTitle = '3D Joint Data'
        self.fig = plt.figure()
        self.fig.canvas.set_window_title(self.strTitle)
        self.fig.set_size_inches((settings.screen_cx * 0.49) / self.fig.dpi, (settings.screen_cy * 0.465) / self.fig.dpi)

        self.fig.canvas.mpl_connect('resize_event', self.onresize)
        self.fig.canvas.mpl_connect('close_event', self.onclose)

        offset=0.08
        self.ax = Axes3D(self.fig, rect=[0, offset, 1.0, 1.0-offset]) # (left, bottom, width, height)
        self.ax.view_init(10, 10)
        self.ax.dist = 9

        self.axSliderTime = plt.axes([0.08, 0.055, 0.80, 0.04], facecolor='lightgoldenrodyellow')
        self.slidertime = Slider(self.axSliderTime, 'Time', 0.0, 1.0, valinit=0, valstep=0.005, valfmt='%0.03fs')
        self.cidSlider = self.slidertime.on_changed(self.onsliderupdate)

        # set currect axes context back to main axes self.ax
        plt.sca(self.ax)

        offset = 0.112
        rect = [0.08, offset + 0.036, 0.80, 0.03]
        self.axFrameBlocks1 = plt.axes(rect)

        rect = [0.08, offset, 0.80, 0.03]
        self.axFrameBlocks2 = plt.axes(rect)

        # set currect axes context back to main axes self.ax
        plt.sca(self.ax)
        
        # add empty joints placeholders
        # [x, y, z, father joint] serial number is the same as Kinect
        self.joints.append([0,0,0,-1])  # spineBase,#0
        self.joints.append([0,0,0,0])   # spineMid,#1
        self.joints.append([0,0,0,20])  # neck,#2
        self.joints.append([0,0,0,2])   # head, #3
        self.joints.append([0,0,0,20])  # shoulderLeft, #4
        self.joints.append([0,0,0,4])   # elbowLeft, #5
        self.joints.append([0,0,0,5])   # wristLeft, #6
        self.joints.append([0,0,0,6])   # handLeft, #7
        self.joints.append([0,0,0,20])  # shoulderRight, #8
        self.joints.append([0,0,0,8])   # elbowRight, #9
        self.joints.append([0,0,0,9])   # wristRight, #10
        self.joints.append([0,0,0,10])  # handRight, #11
        self.joints.append([0,0,0,0])   # hipLeft, #12
        self.joints.append([0,0,0,12])  # kneeLeft, #13
        self.joints.append([0,0,0,13])  # ankleLeft, #14
        self.joints.append([0,0,0,14])  # footLeft, #15
        self.joints.append([0,0,0,0])   # hipRight, #16
        self.joints.append([0,0,0,16])  # kneeRight, #17
        self.joints.append([0,0,0,17])  # ankleRight, #18
        self.joints.append([0,0,0,18])  # footRight, #19
        self.joints.append([0,0,0,1])   # spineSoulder, #20
        self.joints.append([0,0,0,7])   # handTipLeft, #21
        self.joints.append([0,0,0,6])   # thumbLeft, #22
        self.joints.append([0,0,0,11])  # handTipRight, #23
        self.joints.append([0,0,0,10])  # thumbTipRight, #24
        
        # Importing teh T-model and centering its root on 0 to start
        support_dir = Path(filePath).resolve().parents[4]  # Adjust for correct model path
        model_path = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
        
        # Load SMPLH body model
        bm = BodyModel(bm_fname=model_path, num_betas=16).to('cpu')

        # T-pose: zero pose and zero shape
        body_pose = torch.zeros((1, 63))  # (1, 63)
        betas = torch.zeros((1, 16))
        trans = torch.zeros((1, 3))

        # Output: joints and mesh
        out = bm.forward(body_pose=body_pose, betas=betas, transl=trans)

        # Canonical joint positions in T-pose (shape: [1, 52, 3])
        self.amass_joints = out.Jtr[0].detach().cpu().numpy()
        # self.amass_joints[:,  [1, 2]] = self.amass_joints[:,  [2, 1]]  # Y ↔ Z
        
        move = self.amass_joints[0]
        self.amass_joints-=move
        
        # scale, spine_shoulder->spine_midlle is 5
        d = np.linalg.norm(self.amass_joints[3]-self.amass_joints[6])
        self.scale = (3.0 / d)
        self.amass_joints*=self.scale
        
        self.AMASS_TO_KINECT_MAP = {
            "spineB": 0, "spineM": 3,
            "neck": 12, "head": 15,
            "shoulderL": 16, "elbowL": 18, "wristL": 20, "handL": 25,
            "shoulderR": 17, "elbowR": 19, "wristR": 21, "handR": 41,
            "hipL": 1, "kneeL": 4, "ankleL": 7, "footL": 10,
            "hipR": 2, "kneeR": 5, "ankleR": 8, "footR": 11,
            "spineS": 6,"handTL": 34, "thumbL": 35, "handTR": 49, "thumbR": 50
            }

        self.skeleton_connections = [
            ('spineB', 'spineM'), ('spineM', 'spineS'), ('spineS', 'neck'), ('neck', 'head'),
            ('spineS', 'shoulderL'), ('shoulderL', 'elbowL'), ('elbowL', 'wristL'), ('wristL', 'handL'),
            ('spineS', 'shoulderR'), ('shoulderR', 'elbowR'), ('elbowR', 'wristR'), ('wristR', 'handR'),
            ('spineB', 'hipL'), ('hipL', 'kneeL'), ('kneeL', 'ankleL'), ('ankleL', 'footL'),
            ('spineB', 'hipR'), ('hipR', 'kneeR'), ('kneeR', 'ankleR'), ('ankleR', 'footR')
        ]

        # Ordered joint names for drawing: index is your internal joint ID
        self.joint_names_ordered = [
            'spineB', 'spineM', 'neck', 'head',
            'shoulderL', 'elbowL', 'wristL', 'handL',
            'shoulderR', 'elbowR', 'wristR', 'handR',
            'hipL', 'kneeL', 'ankleL', 'footL',
            'hipR', 'kneeR', 'ankleR', 'footR', 'spineS'
        
        ]
        self.name_to_index = {name: i for i, name in enumerate(self.joint_names_ordered)}

        # Get parent index per joint using the connections
        self.parents = [-1] * len(self.joint_names_ordered)
        for parent, child in self.skeleton_connections:
            child_idx = self.name_to_index[child]
            parent_idx = self.name_to_index[parent]
            self.parents[child_idx] = parent_idx


    # -----------------------------------------------------------------------------
    # canvas close event
    #
    def onclose(self, event):
        self.fig = None
        # if user closes this figure, let the main application know and to exit
        settings.application.close()

    #------------------------------------------------------------------------------
    # canvas resize event
    #
    def onresize(self, event):
        self.setAxisLimits()

    #------------------------------------------------------------------------------
    # slider update event
    #
    def onsliderupdate(self, val):
        # map to [0..1]
        p = self.slidertime.val / (self.slidertime.valmax - self.slidertime.valmin)
        self.selectTime(p, False)

        # call main application so that other graphs can be updated as well
        settings.application.selectTime(p, self)

    # -----------------------------------------------------------------------------
    #
    def updateInputName(self):
        self.fig.canvas.set_window_title(self.strTitle + ' - [' + settings.application.strBeautifiedInputFile + ']')

    #------------------------------------------------------------------------------
    #
    def saveView(self):
        if (self.fig is None):
            return

        filePath = os.path.join(settings.application.outputFolder, settings.application.outputName + '_3DJointData.png')
        filePath = settings.checkFileAlreadyExists(filePath, fileExt=".png", fileTypes=[('png files', '.png'), ('all files', '.*')])
        if (filePath is None):
            return

        try:
            self.fig.savefig(filePath, bbox_inches='tight')
            settings.application.logMessage("3D Joint Data view was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving 3D Joint Data view to '" + settings.beautifyPath(filePath) + "': " + str(e))

    #------------------------------------------------------------------------------
    # render frame blocks for both kinect and labanotation
    #
    def renderFrameBlocks(self):
        self.axFrameBlocks1.clear()
        self.axFrameBlocks2.clear()

        # after a clear() annSelection object is gone. reset variable
        self.annSelection = None

        padding = 0.0
        duration = 0.0

        cnt = len(self.jointFrames)
        if (cnt > 0):
            duration = self.jointFrames[cnt-1]['timeS'][0]

            maxTime = (duration / 1000.0)

            def format_func(value, tick_number):
                return r"${:.2f}$".format(value)

            self.axSliderTime.set_xlim((0, maxTime))
            self.axSliderTime.tick_params(axis='x', labelsize=8)
            self.axSliderTime.get_xaxis().set_major_locator(ticker.AutoLocator())
            self.axSliderTime.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

            # show minor ticks every 5 frames and major ticks every 10 frames on the top of frame block 1
            onetick = (maxTime) / float(cnt)
            self.axFrameBlocks1.set_xlim((0, maxTime))
            self.axFrameBlocks1.xaxis.tick_top()
            self.axFrameBlocks1.xaxis.set_minor_locator(ticker.MultipleLocator(onetick * 5))
            self.axFrameBlocks1.xaxis.set_major_locator(ticker.MultipleLocator(onetick * 10))
            self.axFrameBlocks1.set_xticklabels([])
            self.axFrameBlocks1.get_yaxis().set_visible(False)

            self.axFrameBlocks2.set_xlim((0, maxTime))
            self.axFrameBlocks2.get_xaxis().set_visible(False)
            self.axFrameBlocks2.get_yaxis().set_visible(False)

        # render individual kinect joint frame blocks
        xx = self.axFrameBlocks1.get_xlim()
        yy = self.axFrameBlocks1.get_ylim()
        cx = (xx[1] - xx[0])
        cy = (yy[1] - yy[0])
        padding = cx * 0.01
        cnt = len(self.jointFrames)
        if (cnt > 0):
            padding = (cx / cnt) * 0.3
            for i in range(0, cnt):
                start = (self.jointFrames[i]['timeS'][0] / 1000.0) - (padding / 2.0)
                x_width = padding
                isFilled = self.jointFrames[i]['filled'][0]
                c = 'r' if (isFilled) else 'b'
                p = patches.Rectangle((start, yy[0]), x_width, cy, alpha=0.50, color=c)
                self.axFrameBlocks1.add_patch(p)

        self.axFrameBlocks1.text((maxTime), (cy / 2.0), '  Original', horizontalalignment='left', verticalalignment='center', color='black')

        # render individual laban frame blocks
        xx = self.axFrameBlocks2.get_xlim()
        yy = self.axFrameBlocks2.get_ylim()
        cx = (xx[1] - xx[0])
        cy = (yy[1] - yy[0])
        cnt = len(self.timeS)
        if (cnt > 0):
            for i in range(0, cnt):
                start = (self.timeS[i] / 1000.0) - (padding / 2.0)
                x_width = padding
                p = patches.Rectangle((start, yy[0]), x_width, cy, alpha=0.50, color='g')
                self.axFrameBlocks2.add_patch(p)

        self.axFrameBlocks2.text((maxTime), (cy / 2.0), '  Labanotation', horizontalalignment='left', verticalalignment='center', color='black')

        self.fig.canvas.draw_idle()

    #------------------------------------------------------------------------------
    #
    def setJointFrames(self, jointFrames_in):
        self.jointFrames = copy.copy(jointFrames_in)

        if (len(self.jointFrames) > 0):
            timeS0 = self.jointFrames[len(self.jointFrames)-1]['timeS'][0]
            self.slidertime.valmax = float(timeS0) / 1000.0
        else:
            self.slidertime.valmax = 0.0

        # set the slider axes to take valmax change
        self.slidertime.ax.set_xlim(self.slidertime.valmin, self.slidertime.valmax)
        self.fig.canvas.draw_idle()

        self.renderFrameBlocks()

    #------------------------------------------------------------------------------
    #
    def setLabanotation(self, timeS, all_laban):
        self.timeS = copy.copy(timeS)
        self.all_laban = copy.copy(all_laban)

        self.renderFrameBlocks()

    #------------------------------------------------------------------------------
    #
    
    
    def selectTime(self, time, fUpdateSlider=False):
        time = self.slidertime.valmin + (time * (self.slidertime.valmax - self.slidertime.valmin))

        # disconnect slider update callback to avoid endless loop. Reconnect 
        # once slider value reset
        if (fUpdateSlider):
            self.slidertime.disconnect(self.cidSlider)
            self.slidertime.set_val(time)
            self.cidSlider = self.slidertime.on_changed(self.onsliderupdate)

        cnt = len(self.jointFrames)
        if (cnt == 0):
            return

        # find the frame corresponding to the given time
        for idx in range(0,cnt):
            temp = self.jointFrames[idx]
            kt = float(temp['timeS'][0]) / 1000.0
            if (kt >= time):
                break

            self.isInterpolatedKeyFrame = temp['filled'][0]

        # take a kinect snapshot of joints in time and render them.
        # Map to graph's xyz space
        a = self.joints
        for i in range(0, 25):
            a[i][0] = -temp[0][i+2][2]
            a[i][1] = -temp[0][i+2][0]
            a[i][2] =  temp[0][i+2][1]

        self.drawKinectSkeleton()
 
        if ((self.all_laban is not None) and (len(self.all_laban) > 0)):
            cnt = len(self.timeS)

            # find the frame corresponding to the given time
            for idx in range(0, cnt):
                kt = (self.timeS[idx]) / 1000.0
                if (kt >= time):
                    break

            laban = self.all_laban[idx]
            time = int(self.timeS[idx])

            if (settings.fVerbose):
                print('Right Elbow:' + str(elR[t][1]) + ', ' + str(elR[t][2]))
                print('Right Wrist:' + str(wrR[t][1]) + ', ' + str(wrR[t][2]))
                print('Left Elbow:' + str(elL[t][1]) + ', ' + str(elL[t][2]))
                print('Left Wrist:' + str(wrL[t][1]) + ', ' + str(wrL[t][2]))
                print('Right Knee:' + str(knR[t][1]) + ', ' + str(knR[t][2]))
                print('Left Knee:' + str(knL[t][1]) + ', ' + str(knL[t][2]))
                print('Right Ankle:' + str(anR[t][1]) + ', ' + str(anR[t][2]))
                print('Left Ankle:' + str(anL[t][1]) + ', ' + str(anL[t][2]))
                print('Right Foot:' + str(ftR[t][1]) + ', ' + str(ftR[t][2]))
                print('Left Foot:' + str(ftL[t][1]) + ', ' + str(ftL[t][2]))
                print('Head:' + str(head[t][1]) + ', ' + str(head[t][2]))
                print('Torso:' + str(torso[t][1]) + ', ' + str(torso[t][2]))

            curr_laban = [
                        'Start Time:' + str(time),
                        'Duration:0',
                        
                        # Staff-aligned ordering
                        'Left Elbow:' + laban[0][0] + ':' + laban[0][1],
                        'Left Wrist:' + laban[1][0] + ':' + laban[1][1],
                        'Torso:' + laban[2][0] + ':' + laban[2][1],
                        'Left Ankle:' + laban[3][0] + ':' + laban[3][1],
                        'Left Foot:' + laban[4][0] + ':' + laban[4][1],
                        'Left Knee:' + laban[5][0] + ':' + laban[5][1],
                        'Right Knee:' + laban[6][0] + ':' + laban[6][1],
                        'Right Foot:' + laban[7][0] + ':' + laban[7][1],
                        'Right Ankle:' + laban[8][0] + ':' + laban[8][1],
                        'Torso_Repeat:' + laban[9][0] + ':' + laban[9][1],
                        'Right Wrist:' + laban[10][0] + ':' + laban[10][1],
                        'Right Elbow:' + laban[11][0] + ':' + laban[11][1],
                        'Head:' + laban[12][0] + ':' + laban[12][1],
                        'Support:' + laban[13][1],
                        'Rotation:ToLeft:' + str(laban[14])
                    ]



            self.drawLabanotationSkeleton(laban = curr_laban)

            if (self.axFrameBlocks2 != None):
                if (self.annSelection != None):
                     self.annSelection.remove()

                color = 'green'
                time = time / 100.0
                self.annSelection = self.axFrameBlocks2.annotate('', xy=(time, 0.0), xytext=(time, -0.5),
                    weight='bold', color=color,
                    arrowprops=dict(arrowstyle='wedge', connectionstyle="arc3", color=color))

    #------------------------------------------------------------------------------
    #
    def setAxisLimits(self):
        # get canvas size in pixels
        size = self.fig.get_size_inches() * self.fig.dpi

        # calculate axis limits while keeping aspect ratio at 1
        aspect = size[0] / size[1];
        min = -20
        max = 20
        axmin = (min) if (aspect < 1) else (min * aspect)
        axmax = (max) if (aspect < 1) else (max * aspect)
        aymin = (min) if (aspect > 1) else (min / aspect)
        aymax = (max) if (aspect > 1) else (max / aspect)

        # set limits. Remember our axis mapped (x=z), (y=x) and (z=y)
        self.ax.set_xlim3d(min, max)
        self.ax.set_ylim3d(axmin, axmax)
        self.ax.set_zlim3d(aymin, aymax)

    #------------------------------------------------------------------------------
    #
    def resetGraph(self):
        # clear canvas
        self.ax.clear()

        self.setAxisLimits()
        
        # Draw x, y, and z axis markers in the same way you were in
        # the code snippet in your question...
        xspan, yspan, zspan = 3 * [np.linspace(0, 6, 15)]
        zero = np.zeros_like(xspan)

        self.ax.plot3D(xspan, zero, zero, 'k--', gid='axis')
        self.ax.plot3D(zero, yspan, zero, 'k--', gid='axis')
        self.ax.plot3D(zero, zero, zspan, 'k--', gid='axis')

        q = -.20
        w = 1
        self.ax.text(xspan.max() + w, q, q, "z", gid='axis', color='black')
        self.ax.text(q, yspan.max() + w, q, "x", gid='axis', color='black')
        self.ax.text(q, q, zspan.max() + w, "y", gid='axis', color='black')

        self.ax.xaxis._axinfo['tick']['color'] = (1.0, 1.0, 1.0, 0.5)
        self.ax.yaxis._axinfo['tick']['color'] = (1.0, 1.0, 1.0, 0.5)
        self.ax.zaxis._axinfo['tick']['color'] = (1.0, 1.0, 1.0, 0.5)

        self.ax.xaxis._axinfo['axisline']['color'] = (1.0, 1.0, 1.0, 0.5)
        self.ax.yaxis._axinfo['axisline']['color'] = (1.0, 1.0, 1.0, 0.5)
        self.ax.zaxis._axinfo['axisline']['color'] = (1.0, 1.0, 1.0, 0.5)

        # get rid of the panes                          
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5)) 
        # self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

        # get rid of the spines                         
        #self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        #self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.5)) 
        #self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.5))

        # don't show tick labels
        self.ax.xaxis.set_major_formatter(plt.NullFormatter())
        self.ax.yaxis.set_major_formatter(plt.NullFormatter())
        self.ax.zaxis.set_major_formatter(plt.NullFormatter())

    #------------------------------------------------------------------------------
    #
    #   rotate around x axis:   rotate around y axis:   rotates around
    #    np.array([              np.array([              np.array([
    #        [1,  0,  0],            [c,  0, -s],            [c, -s,  0],
    #        [0,  c, -s],            [0,  1,  0],            [s,  c,  0],
    #        [0,  s,  c]             [s,  0,  c]             [0,  0,  1],
    #      ]))                     ]))                     ]))

    def correctSkeletonRotation(self):
        shL = np.zeros(3)
        shR = np.zeros(3)
        spM = np.zeros(3)
        shL[0] = self.joints[4][1]  # left shoulder
        shL[1] = self.joints[4][2]
        shL[2] = self.joints[4][0]
        shR[0] = self.joints[8][1]  # right shoulder
        shR[1] = self.joints[8][2]
        shR[2] = self.joints[8][0]
        spM[0] = self.joints[1][1]  # spine middle
        spM[1] = self.joints[1][2]
        spM[2] = self.joints[1][0]
        
        # find where the performer is facing, rotate the body figure
        sh = np.zeros((3,3))
        v1 = shL-shR
        v2 = spM-shR
        sh[0] = np.cross(v2,v1) # x axis
        x = sh[0][0]
        y = sh[0][1]
        z = sh[0][2]

        # rotate around y axis
        r = math.sqrt(z*z+x*x)
        sinth = x/r
        costh = z/r
        conv = np.zeros((3,3))
        conv[0][0] = costh
        conv[0][1] = 0
        conv[0][2] = -sinth
        conv[1][0] = 0
        conv[1][1] = 1
        conv[1][2] = 0
        conv[2][0] = sinth
        conv[2][1] = 0
        conv[2][2] = costh
        
        for i in range(len(self.joints)):
            tmp_in = np.zeros(3)
            tmp_in[0] = self.joints[i][1]
            tmp_in[1] = self.joints[i][2]
            tmp_in[2] = self.joints[i][0]
            tmp_out = np.dot(conv,tmp_in)
            self.joints[i][1] = tmp_out[0]
            self.joints[i][2] = tmp_out[1]
            self.joints[i][0] = tmp_out[2]


    #------------------------------------------------------------------------------
    #
    def draw_joints(self, c, a = 1.0):
        alpha = a
        cnt = len(self.joints) - 4 # do not draw hand tips and thumbs
        for i in range(cnt):
            if (i == 7) or (i == 11): # do not draw hands
                continue

            if (i==20) or (i==4) or (i==5) or (i==6) or (i==8) or (i==9) or (i==10):
                alpha = a
            else:
                alpha = 0.5

            p = self.joints[i]
            self.ax.scatter(p[0],p[1],p[2],color=c, alpha=alpha)

    #------------------------------------------------------------------------------
    #
    def draw_limbs(self):
        for i in range(1, len(self.joints) - 4):
            if (i == 7) or (i == 11): # do not draw hands
                continue

            if (i==4) or (i==5) or (i==6) or (i==8) or (i==9) or (i==10):
                alpha = 1.0
            else:
                alpha = 0.3

            a = self.joints[i]
            start = [a[0],a[1],a[2]]
            b = self.joints[self.joints[i][3]]
            end = [b[0],b[1],b[2]]
            self.ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'--', color='k', alpha=alpha, gid='limbs')

        # body frame
        a = self.joints[1]
        start = [a[0],a[1],a[2]]
        b = self.joints[4]
        end = [b[0],b[1],b[2]]
        self.ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'--', color='k', alpha=0.3, gid='limbs')
        b = self.joints[8]
        end = [b[0],b[1],b[2]]
        self.ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'--', color='k', alpha=0.3, gid='limbs')
    
    #------------------------------------------------------------------------------
    #
    def drawKinectSkeleton(self):
        self.resetGraph()

        # center skeleton around its spineB
        move_0 = 0#self.joints[0][0]
        move_1 = 0#self.joints[0][1]
        move_2 = 0#self.joints[0][2]

        for i in range(0,len(self.joints)):
            self.joints[i][0] = self.joints[i][0]-move_0
            self.joints[i][1] = self.joints[i][1]-move_1
            self.joints[i][2] = self.joints[i][2]-move_2

        # # rotate skeleton to face forward towards camera
        # self.correctSkeletonRotation()

        # scale, spine_shoulder->spine_midlle is 5
        d0 = self.joints[20][0]-self.joints[1][0]
        d1 = self.joints[20][1]-self.joints[1][1]
        d2 = self.joints[20][2]-self.joints[1][2]

        d = (d0**2+d1**2+d2**2)**0.5

        self.scale = (3.0 / d)

        for i in range(0, len(self.joints)):
            self.joints[i][0] = (self.joints[i][0] * self.scale)
            self.joints[i][1] = (self.joints[i][1] * self.scale) - 15      # offset from center
            self.joints[i][2] = (self.joints[i][2] * self.scale)
            
        faceColor = 'r' if self.isInterpolatedKeyFrame else 'b'

        self.draw_joints(faceColor, 0.50)
        self.draw_limbs()

        # skeleton title
        self.ax.text(self.joints[3][0], self.joints[3][1], self.joints[3][2] + 2, 'Original',
                bbox={'facecolor':faceColor,'alpha':0.50,'edgecolor':'gray','pad':3.5},
                ha='center', va='bottom', color='w') 

        self.fig.canvas.draw_idle()

    #------------------------------------------------------------------------------
    # convert labanotation to joint points
    #
    def mapLabanotation2Joints(self, laban):
        """
        Builds self.joints using canonical AMASS T-pose + laban offsets.
        self.joints[i] = [x, y, z, parent_idx]
        """

        # Reset joint positions
        # for i in range(len(self.joint_names_ordered)):
        #     self.joints[i][0:3] = [0, 0, 0]
            # self.joints[i][3] = self.parents[i]

        # Set pelvis (spineB, index 0)
        
        root_name = self.joint_names_ordered[0]
        root_amass_idx = self.AMASS_TO_KINECT_MAP[root_name]
        self.joints[0][0:3] = self.amass_joints[root_amass_idx]
        
        footL_y = self.amass_joints[self.AMASS_TO_KINECT_MAP["ankleR"]][1]
        footR_y = self.amass_joints[self.AMASS_TO_KINECT_MAP["ankleL"]][1]
        base_foot=min(footL_y, footR_y)
        # Joints with laban overrides (mapped by index in self.joints)
        laban_override_map = {20: "torso",        # spineMid -> spineShoulder
                                3: "head",          # neck -> head
                                5: "left elbow",    # shoulderLeft -> elbowLeft
                                6: "left wrist",    # elbowLeft -> wristLeft
                                9: "right elbow",   # shoulderRight -> elbowRight
                                10: "right wrist",  # elbowRight -> wristRight
                                13: "left knee",    # hipLeft -> kneeLeft
                                14: "left ankle",   # kneeLeft -> ankleLeft
                                15: "left foot",    # ankleLeft -> footLeft
                                17: "right knee",   # hipRight -> kneeRight
                                18: "right ankle",  # kneeRight -> ankleRight
                                19: "right foot",   # ankleRight -> footRight
                            }
        for i in [1,20]+list(range(2, 20)):
            parent_idx = self.joints[i][3]
            joint_name = self.joint_names_ordered[i]
            amass_idx = self.AMASS_TO_KINECT_MAP[joint_name]

            if parent_idx == -1:
                continue  # Skip root, already placed

            parent_pos = self.joints[parent_idx][0:3]
            
            parent_name = self.joint_names_ordered[parent_idx]
            parent_amass_idx = self.AMASS_TO_KINECT_MAP[parent_name]
            
            limb_relative = self.amass_joints[amass_idx] - self.amass_joints[parent_amass_idx]
            if i in laban_override_map.keys():
                joint_name_laban= laban_override_map[i]
                limb_lenght=np.linalg.norm(limb_relative)
                vec = self.laban2vec(laban, joint_name_laban, limb_lenght)
                self.joints[i][0:3] = parent_pos + np.array(vec)
            elif i in [7,11,21,22,23,24]:
                
                joint_name_laban= laban_override_map[parent_idx]
                limb_lenght=np.linalg.norm(limb_relative)
                vec = self.laban2vec(laban, joint_name_laban, limb_lenght)
                self.joints[i][0:3] = parent_pos + np.array(vec)
            else:
               
                self.joints[i][0:3] = parent_pos +  limb_relative
            
            
        min_foot= min(self.joints[14][1], self.joints[18][1])
        foot_offset=base_foot-min_foot
        if laban[15].split(":")[1]=="Jump":
            for joint in self.joints:
                joint[1]+=self.scale*0.1
        else:
            for joint in self.joints:
                joint[1]+=foot_offset
           
    #------------------------------------------------------------------------------
    # calculate the given joint using its parent joint and a vector
    #
    def calc_joint(self, target, vec):
        self.joints[target][0] = self.joints[self.joints[target][3]][0] + vec[0]
        self.joints[target][1] = self.joints[self.joints[target][3]][1] + vec[1]
        self.joints[target][2] = self.joints[self.joints[target][3]][2] + vec[2]

    #------------------------------------------------------------------------------
    # convert the labanotation for a given limb to a vector
    def laban2vec(self, laban, limb, limb_length=5):
        theta = 175
        phi = 0
        pi = 3.1415926
        
        support=False
        for i in range(len(laban)):
            laban[i] = laban[i].lower()
            tmp = laban[i].split(":")
            if tmp[0] == limb:
                if " o" in tmp[2]:
                    support=True
                dire = tmp[1]
                level = tmp[2]  # remove support marker
                if level == "high":
                    theta = 45
                elif level == "normal":
                    theta = 90
                elif level == "low":
                    theta = 135
                elif level == "high o":
                    theta = 170
                elif level == "normal o":
                    theta = 135
                elif level == "low o":
                    theta = 90
                # elif dire=="place" and not (i in [7,8]):
                #     theta = 180 if "low" in level else 5 
                else:
                    theta = 180
                    print('Unknown Level.')
                    
                
                if dire == "forward":
                    phi = 0
                elif dire == "right forward":
                    phi = -45
                elif dire == "right":
                    phi = -90
                elif dire == "right backward":
                    phi = -135
                elif dire == "backward":
                    phi = 180
                elif dire == "left backward":
                    phi = 135
                elif dire == "left":
                    phi = 90
                elif dire == "left forward":
                    phi = 45
                elif dire=="place":
                    phi = 0
                    if not (i in [7,8]):
                        theta = 180 if "low" in level else 5 
                else:
                    phi = 0
                    print('Unknown Direction.')
                break
        
        y = limb_length * math.cos(math.radians(theta))
        x = limb_length * math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        z = limb_length * math.sin(math.radians(theta)) * math.cos(math.radians(phi))

        return [x, y, z]






    #------------------------------------------------------------------------------
    #
    def drawLabanotationSkeleton(self, laban=""):
        if (laban == ''):
            return

        for i in range(0,len(self.joints)):
            self.joints[i][0] = 0
            self.joints[i][1] = 0
            self.joints[i][2] = 0

        self.mapLabanotation2Joints(laban)

        # map to graph xyz
        for i in range(0,len(self.joints)):
            x = self.joints[i][0]
            y = self.joints[i][1]
            z = self.joints[i][2]

            self.joints[i][0] = z
            self.joints[i][1] = x + 15      # offset from center
            self.joints[i][2] = y

        self.draw_joints('g', 0.50)
        self.draw_limbs()

        # skeleton title
        self.ax.text(self.joints[3][0], self.joints[3][1], self.joints[3][2] + 2, 'Labanotation',
                bbox={'facecolor':'g','alpha':0.50,'edgecolor':'gray','pad':3.5},
                ha='center', va='bottom', color='w') 
        
        self.fig.canvas.draw_idle()


    def calculate_ecm(self, keyframes):
        
        original_joints=np.zeros((len(keyframes),25, 3))
        laban_joints=original_joints.copy()
        
        for i, idx in enumerate(keyframes):
            
            temp=self.jointFrames[idx][0]
             # scale, spine_shoulder->spine_midlle is 5
            d0 = temp[22][0]-temp[3][0]
            d1 = temp[22][1]-temp[3][1]
            d2 = temp[22][2]-temp[3][2]

            # d = (d0**2+d1**2+d2**2)**0.5

            # scale = (3.0 / d)

            for k in range(0, 25):
                original_joints[i,k,0] = temp[k+2][0]#*scale
                original_joints[i,k,1] = temp[k+2][1]#*scale
                original_joints[i,k,2] = -temp[k+2][2]#*scale
            laban = self.all_laban[idx]

            curr_laban =  [
                        'Start Time:' + str(0),
                        'Duration:0',
                    # Staff-aligned ordering
                    'Left Elbow:' + laban[0][0] + ':' + laban[0][1],
                    'Left Wrist:' + laban[1][0] + ':' + laban[1][1],
                    'Torso:' + laban[2][0] + ':' + laban[2][1],
                    'Left Ankle:' + laban[3][0] + ':' + laban[3][1],
                    'Left Foot:' + laban[4][0] + ':' + laban[4][1],
                    'Left Knee:' + laban[5][0] + ':' + laban[5][1],
                    'Right Knee:' + laban[6][0] + ':' + laban[6][1],
                    'Right Foot:' + laban[7][0] + ':' + laban[7][1],
                    'Right Ankle:' + laban[8][0] + ':' + laban[8][1],
                    'Torso_Repeat:' + laban[9][0] + ':' + laban[9][1],
                    'Right Wrist:' + laban[10][0] + ':' + laban[10][1],
                    'Right Elbow:' + laban[11][0] + ':' + laban[11][1],
                    'Head:' + laban[12][0] + ':' + laban[12][1],
                    'Support:' + laban[13][1],
                    'Rotation:ToLeft:' + str(laban[14])
                ]

            # take a kinect snapshot of joints in time and render them.
            # Map to graph's xyz space
            self.mapLabanotation2Joints(curr_laban)
            laban_joints[i]=np.array(self.joints)[:,:3]
            
        
       
        original = np.array(original_joints)
        laban_joints/=self.scale
        
        # Promediado sobre frames y ejes (x,y,z)
        mse_per_joint = np.mean((original - laban_joints)[:,:21] ** 2, axis=(0, 2)) 
        print(mse_per_joint)
        
        
        def plot_skeletons(original_joints, laban_joints, parents):
            fig = plt.figure(figsize=(12, 6))
            
            # Left: Original
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title("Original Skeleton (Keyframe 0)")
            plot_skeleton(ax1, original_joints, parents, color='blue')

            # Right: Reconstructed
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("Laban Reconstruction (Keyframe 0)")
            plot_skeleton(ax2, laban_joints, parents, color='green')

            plt.tight_layout()
            plt.show()

        def plot_skeleton(ax, joints, parents, color='blue'):
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=30)
            for i in range(len(joints)):
                p = parents[i]
                if p == -1:
                    continue
                x = [joints[i][0], joints[p][0]]
                y = [joints[i][1], joints[p][1]]
                z = [joints[i][2], joints[p][2]]
                ax.plot(x, y, z, color=color, linewidth=2)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-20, 20)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=15, azim=-70)
        
        # Assume you're using the same skeleton definition for both
        parents = [j[3] for j in self.joints[:21]]
        plot_skeletons(original_joints[2,:21,:], laban_joints[2,:21,:], parents)

        return mse_per_joint, laban_joints  

    #------------------------------------------------------------------------------
    #
