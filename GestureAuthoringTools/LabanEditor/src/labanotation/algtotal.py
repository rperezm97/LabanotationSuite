# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys, os, math, copy
import json
from math import sin, cos, sqrt, radians

import numpy as np
from decimal import Decimal
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, Cursor, Button
from matplotlib.backend_bases import MouseEvent
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from tqdm import tqdm 

try:
    from tkinter import messagebox
except ImportError:
    # Python 2
    import tkMessageBox as messagebox

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tool'))

import settings

from . import labanProcessor as lp

import kp_extractor as kpex

import accessory as ac
import wavfilter as wf
import cluster as cl



from . import physical_indices as physics
from . import kbs_laban  as kbs_laban
AMASS_TO_KINECT_MAP = {
    "spineB": 0, "spineM": 3, "spineS": 6, "neck": 12, "head": 15,
    "shoulderL": 16, "elbowL": 18, "wristL": 20, "handL": 25,
    "shoulderR": 17, "elbowR": 19, "wristR": 21, "handR": 41,
    "hipL": 1, "kneeL": 4, "ankleL": 7, "footL": 10,
    "hipR": 2, "kneeR": 5, "ankleR": 8, "footR": 11,
    "handTL": 34, "thumbL": 35, "handTR": 49, "thumbR": 50
}

class Algorithm:
    algorithm = None
    ax = None

    jointFrames = []

    timeS = None
    all_laban = []

    unfilteredTimeS = None
    unfilteredLaban = []

    labandata = OrderedDict()

    line_ene = None
    vlines = None
    y_data = []
    points = []

    data_fps = 120
    dragging_sb = False
    dragging_point = None

    selectedFrame = 0
    selectedFrameMarker = None

    default_gauss_window_size = 61
    default_gauss_sigma = 5

    gauss_window_size = default_gauss_window_size
    gauss_sigma = default_gauss_sigma

    STEP_THRESHOLD = 0.2  # Minimum displacement to be a step (meters)
    JUMP_THRESHOLD = 0.2  # Minimum vertical displacement for jumps (meters)
    ROTATION_THRESHOLD = 15  # Minimum rotation for a turn (degrees)

    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self, algorithm):
        self.algorithm = algorithm

    #------------------------------------------------------------------------------
    # reset class variables
    #
    def reset(self):
        self.jointFrames = []

        self.timeS = None
        self.all_laban = []

        self.unfilteredTimeS = None
        self.unfilteredLaban = []

        self.labandata = OrderedDict()

        self.line_ene = None
        self.vlines = None

        self.y_data = []
        self.points = []

        self.data_fps = 120
        self.dragging_sb = False
        self.dragging_point = None

        self.selectedFrame = 0
        self.selectedFrameMarker = None

    #------------------------------------------------------------------------------
    # convert joint data frames to labanotation
    #
    def convertToLabanotation(self, ax, jointD, forceReset,
                              base_rotation_style='every'):
        if (forceReset):
            self.reset()

        self.ax = ax

        self.jointFrames = copy.copy(jointD[0])

        cnt = len( self.jointFrames)

        self.data_fps = 120
        self.duration =  self.jointFrames[cnt-1]['timeS'][0] if (cnt > 0) else 0.0

        # clear canvas
        if (self.ax != None):
            self.ax.clear()
            self.selectedFrameMarker = None

            self.line_ene = None
            self.vlines = None
            self.y_data = []
            self.points = []
        # fps=120
        # self.unfilteredTimeS= [int(1 + (idx * (1000 / fps))) for idx in range(cnt)]
        
        self.calculateUnfilteredLaban(base_rotation_style=base_rotation_style)
        self.timeS, self.all_laban, self.keyframes=self.totalEnergy(jointD[1])

        self.all_laban = kbs_laban.run_classification(settings.application.outputFilePathOwl, jointD[1:], self.keyframes)

        # return self.totalEnergy()

        
        # # self.calculateUnfilteredLaban(base_rotation_style=base_rotation_style)
      
        # self.unfilteredLaban= kbs_laban.run_KBS(jointD,self.keyframes)
       
        return self.timeS,self.all_laban, [0]+self.keyframes+[len(jointD[1])-1]

    
    def calculateUnfilteredLaban(self, base_rotation_style='every'):
        """
        Computes Labanotation symbols for **arms, legs, support, and rotation**.

        Args:
            base_rotation_style (str): 'first' → uses first frame's base rotation;
                                    'every' → recalculates it per frame.
        """
        base_rotation = None
        base_translation = None

        if base_rotation_style == 'first':
            try:
                base_rotation = self.jointFrames[0]["R"][0]
                base_translation = self.jointFrames[0]["T"][0]
            except KeyError:
                base_rotation = lp.calculate_base_rotation(self.jointFrames[0])
                base_translation = None

        cnt = len(self.jointFrames)


        # get hand position
        self.unfilteredTimeS = np.zeros(cnt)
       
        # ✅ Store spherical coordinate data for joints
        elR = np.zeros((cnt, 3))
        elL = np.zeros((cnt, 3))
        wrR = np.zeros((cnt, 3))
        wrL = np.zeros((cnt, 3))
        knR = np.zeros((cnt, 3))
        knL = np.zeros((cnt, 3))
        anR = np.zeros((cnt, 3))
        anL = np.zeros((cnt, 3))
        fL= np.zeros((cnt, 3))
        fR= np.zeros((cnt, 3))
        head = np.zeros((cnt, 3))
        
        shR = np.zeros((cnt, 3))
        shL = np.zeros((cnt, 3))
        torso = np.zeros((cnt, 3))

        # ✅ Store support (steps, jumps, turns)
        support = np.full(cnt, 'Stable', dtype=object)  # Default: stable support
        base_rotation_partial=self.jointFrames[0]["T"][0]
        base_translation_partial=self.jointFrames[0]["R"][0]
        
        
        for i in tqdm(range(cnt), desc="Processing Frames"):
            if base_rotation_style == 'every':
                try:
                    base_rotation = self.jointFrames[i]["R"][0]
                    base_translation = self.jointFrames[i]["T"][0]
                except KeyError:
                    base_rotation = lp.calculate_base_rotation(self.jointFrames[i])
                    base_translation = None

            self.unfilteredTimeS[i] = self.jointFrames[i]['timeS'][0]

            # ✅ Convert joints to spherical coordinates
            (elR[i], elL[i], wrR[i], wrL[i], 
            knR[i], knL[i], anR[i], anL[i], fR[i],fL[i], head[i], torso[i],shR[i], shL[i]) = lp.raw2sphere(
                self.jointFrames[i], base_rotation=base_rotation, base_translation=base_translation
            )
            if i==0:
                print(lp.raw2sphere(
                self.jointFrames[i], base_rotation=base_rotation, base_translation=base_translation
                 ))

        # ✅ Convert to Labanotation
        self.unfilteredLaban = []
        footL_y = self.jointFrames[0]["footL"][0][1]
        footR_y = self.jointFrames[0]["footR"][0][1]
        base_foot=min(footL_y, footR_y)
        #Todo modify ankels and foots
        for i in range(cnt):
            support_type, rotation, base_translation_partial, base_rotation_partial = lp.detect_weight_support(self.jointFrames, i, base_translation_partial, base_rotation_partial, base_foot)
            # ✅ Convert Joints to Labanotation
            self.unfilteredLaban.append([
            lp.coordinate2laban(elL[i][1], elL[i][2], 'arm'),
            lp.coordinate2laban(wrL[i][1], wrL[i][2], 'arm'),
            
            lp.coordinate2laban(torso[i][1], torso[i][2], 'body'),#Todo shoulder
            
            lp.coordinate2laban(anL[i][1], anL[i][2], 'leg'),
            lp.coordinate2laban(fL[i][1], fL[i][2], 'foot'),
            lp.coordinate2laban(knL[i][1], knL[i][2], 'support', support_type[:2], support_type[2]!="Right" and support_type[1]!="Jump") ,

            lp.coordinate2laban(knR[i][1], knR[i][2], 'support', support_type[:2], support_type[2]!="Left" and support_type[1]!="Jump",),
            lp.coordinate2laban(fR[i][1], fR[i][2], 'foot'),
            lp.coordinate2laban(anR[i][1], anR[i][2], 'leg'),
            
            lp.coordinate2laban(torso[i][1], torso[i][2], 'body'),#Todo shoulder
            
            lp.coordinate2laban(wrR[i][1], wrR[i][2], 'arm'),
            lp.coordinate2laban(elR[i][1], elR[i][2], 'arm'),
            
            lp.coordinate2laban(head[i][1], head[i][2], 'head'),
            support_type,
            rotation, 
            base_translation_partial, base_rotation_partial
            ])
        
    #------------------------------------------------------------------------------
    # apply total energy algoritm to joint data frames and calculate labanotation
    #
    def totalEnergy(self, joint_vector):
        """
        Compute aggregated energy from smoothed joint trajectories and extract keyframe indices.

        Args:
            joint_vector (np.ndarray): Array of shape (T, J, 3) with raw joint positions.

        Returns:
            List[int]: Keyframe indices detected as local maxima of the combined energy.
        """
        # 1️⃣ Smooth raw joint positions along the temporal axis
        #    joint_vector shape: (T, J, 3)

        T, J, _ = joint_vector.shape

        # 1️⃣ Smooth at large sigma
        sm_large = physics.smooth_positions(joint_vector,
                                            sigma=self.default_gauss_sigma,
                                            window_size=self.default_gauss_window_size,
                                            )
        # 1a️⃣ Smooth at small sigma (naive)
        sm_small = physics.smooth_positions(joint_vector, 
                                            sigma=1.0, 
                                            window_size=self.default_gauss_window_size)

        dt = 1.0 / self.data_fps
        # 2️⃣ Vel/Acc for both
        vel_L, vel_S = {}, {}
        acc_L, acc_S = {}, {}
        for name in ('wristL','wristR','ankleL','ankleR','head'):
            idx = AMASS_TO_KINECT_MAP[name]
            vel_L[name] = (sm_large[2:, idx] - sm_large[:-2, idx]) / (2*dt)
            acc_L[name] = (sm_large[2:, idx] - 2*sm_large[1:-1, idx] + sm_large[:-2, idx]) / (dt*dt)
            vel_S[name] = (sm_small[2:, idx] - sm_small[:-2, idx]) / (2*dt)
            acc_S[name] = (sm_small[2:, idx] - 2*sm_small[1:-1, idx] + sm_small[:-2, idx]) / (dt*dt)

        # 3️⃣ Compute per-part energy (IJCV)
        e_hands   = kpex.energy_function_ijcv(vel_L['wristL'], acc_L['wristL'], vel_L['wristR'], acc_L['wristR'])
        e_feet    = kpex.energy_function_ijcv(vel_L['ankleL'], acc_L['ankleL'], vel_L['ankleR'], acc_L['ankleR'])
        e_head    = kpex.energy_function_ijcv(vel_L['head'],    acc_L['head'],    vel_L['head'],    acc_L['head']) * 0.5
        e_comb_L  = e_hands + e_feet + e_head
        # naive
        e_hands_s = kpex.energy_function_ijcv(vel_S['wristL'], acc_S['wristL'], vel_S['wristR'], acc_S['wristR'])
        e_feet_s  = kpex.energy_function_ijcv(vel_S['ankleL'], acc_S['ankleL'], vel_S['ankleR'], acc_S['ankleR'])
        e_head_s  = kpex.energy_function_ijcv(vel_S['head'],    acc_S['head'],    vel_S['head'],    acc_S['head']) * 0.5
        e_comb_S  = e_hands_s + e_feet_s + e_head_s

        # 4️⃣ Normalize large‐sigma combined
        ec = e_comb_L
        ec = (ec - ec.min()) / (ec.max() - ec.min() + 1e-8)

        # 5️⃣ Keyframes via gaussian_pecdec
        kf = kpex.gaussian_pecdec(ec)

        # 6️⃣ Inflection points
        infl = ac.inflection(ec)

        # 7️⃣ Save for downstream
        self.y_data = {
            'hands_large': e_hands,
            'feet_large':  e_feet,
            'head_large':  e_head,
            'combined':    ec,
            'naive':       e_comb_S
        }
        self.points = {i: ec[i] for i in kf}
        self.keyframes = kf
        # 8️⃣ Plot
        if (self.ax != None):
            xmax = max(self.unfilteredTimeS) / 1000.0

            self.ax.plot(e_comb_L, color='dimgray', label='Total')
            self.ax.plot(e_comb_S, color='mediumpurple', label='Naive')
       
            self.ax.set_xlim((0, len(e_comb_L)-1))
            self.ax.set_ylim((min(e_comb_L)-0.5, max(e_comb_L)+0.5))

            def format_func(value, tick_number):
                cnt = len(self.unfilteredTimeS)
                idx = int(value)
                if (idx < 0) or (idx >= cnt):
                    return ""

                time = self.unfilteredTimeS[idx] / 1000.0

                return r"${:.2f}$".format(time)

            # look at https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html for fine-tuning ticks
            self.ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

            self.ax.tick_params(axis='y', labelsize=8)

            legend_elements = [Line2D([0], [0], color='dimgray', label='Energy'),
                               Line2D([0], [0], color='mediumpurple', label='Naive Energy'),
                               Patch(facecolor='wheat', edgecolor='wheat', alpha=0.4, label='Labanotation Frame Blocks'),
                               Patch(facecolor='tan', edgecolor='tan', alpha=0.4, label='Labanotation Transition Block'),
                               Line2D([0], [0], marker='o', color='w', label='Peaks', markerfacecolor='slategrey', markersize=10),
                               Line2D([0], [0], marker='o', color='w', label='Inflection', markerfacecolor='k', markersize=10),
                               Line2D([0], [0], marker='*', color='w', label='Labanotation Key Frames', markerfacecolor='g', markersize=16)]

            self.ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1), loc=3, ncol=7) # , mode='expand', borderaxespad=0)
        
        print(kf)
        self.updateEnergyPlotAndLabanScore(True)
        self.highlightLabanotationRegions(self.unfilteredLaban, (min(e_comb_L)-0.5, max(e_comb_L)+0.5))
        # additional energy markers
        if (self.ax != None):
            corner,_,_ = cl.peak_dect(e_comb_L, y_thres=0)
            self.ax.plot(e_comb_L, '.', color = 'slategrey', mew=3, markersize=14, markevery=corner) # bottom

            infl = ac.inflection(e_comb_L)
            self.ax.plot(e_comb_L, '.', color = 'k', mew=3, markersize=12, markevery=infl)

        self.setSelectedFrameMarker()

        return (self.timeS, self.all_laban, self.keyframes)

    #------------------------------------------------------------------------------
    # plot different colors for each labanotation region.
    #
    def highlightLabanotationRegions(self, laban, y):
        if self.ax is None:
            return

        # Split laban into segments (using your existing splitting logic)
        laban_sect = ac.split(laban)
        cnt = len(laban)

        # Compute keyframes from the aggregated energy function
        # indices = kpex.gaussian_pecdec(self.y_data['combined'])
        indices= self.keyframes
        # For each laban segment, choose a color based on whether the start is in the aggregated keyframes.
        for i in range(len(self.jointFrames)):
            start = i
            end = i+1

            c = 'wheat' if start not in indices else 'tan'
            a = 0.4

            # Determine width: for the last segment, extend to the end of the sequence
            x_width = end - start + 0.5 if i < len(laban_sect) - 1 else cnt - start + 0.25

            p = patches.Rectangle((start - 0.25, y[0]), x_width, y[1] - y[0], alpha=a, color=c)
            self.ax.add_patch(p)


    #------------------------------------------------------------------------------
    #
    def getLabanotationKeyframeData(self, idx, time, dur, laban):
        """
        Generates a structured Labanotation keyframe data dictionary for full-body motion.
        """
        data = OrderedDict()
        data["start time"] = [str(time/120)]
        data["duration"] = [str(dur)]

        # ➤ Extract motion direction and level for each body part (following staff order)
        data["left elbow"]   = [laban[0][0], laban[0][1]]
        data["left wrist"]   = [laban[1][0], laban[1][1]]
        data["left body"]    = [laban[2][0], laban[2][1]]
        data["left ankle"]   = [laban[3][0], laban[3][1]]
        data["left foot"]    = [laban[4][0], laban[4][1]]
        data["left knee"]    = [laban[5][0], laban[5][1]]
        data["right knee"]   = [laban[6][0], laban[6][1]]
        data["right foot"]   = [laban[7][0], laban[7][1]]
        data["right ankle"]  = [laban[8][0], laban[8][1]]
        data["right body"] =   [laban[9][0], laban[9][1]]  # Optional: torso again?
        data["right wrist"]  = [laban[10][0], laban[10][1]]
        data["right elbow"]  = [laban[11][0], laban[11][1]]
        data["head"]         = [laban[12][0], laban[12][1]]
        data["support"]      = [laban[13]]
        data["rotation"]     = ['ToLeft', laban[14]]

        return data

    #------------------------------------------------------------------------------
    # update labanotation key frames
    def updateLaban(self, indices):
        self.labandata = OrderedDict()
        positions = []

        self.timeS = []
        self.all_laban = []

        # Ensure indices come from all energy functions
        all_indices = self.keyframes#kpex.gaussian_pecdec(self.y_data['combined'])

        idx = 0
        cnt = len(all_indices)

        if cnt == 0:
            return

        for i in range(cnt):
            j = all_indices[i]

            if ((i==0) and (j != i)):
                time = int(self.unfilteredTimeS[i])
                dur = 1

                # store new time and laban
                self.timeS.append(time)
                self.all_laban.append(self.unfilteredLaban[i])

                positions.append("Position"+str(i))
                self.labandata[positions[idx]] = self.getLabanotationKeyframeData(idx, time, dur, self.unfilteredLaban[i])
                idx = idx + 1
            
            time = int(self.unfilteredTimeS[j])
            dur = '-1' if j == (cnt - 1) else '1'

            # Store new time and laban
            self.timeS.append(time)
            self.all_laban.append(self.unfilteredLaban[j])

            positions.append("Position" + str(i))
            self.labandata[positions[idx]] = self.getLabanotationKeyframeData(idx, time, dur, self.unfilteredLaban[j])
            idx += 1
            
            
          # add a final labanotation keyframe
        i = len(self.unfilteredLaban) - 1
        j = all_indices[cnt - 1]
        if (j != i):
            time = int(self.unfilteredTimeS[i])
            dur = '-1'

            # store new time and laban
            self.timeS.append(time)
            self.all_laban.append(self.unfilteredLaban[i])

            positions.append("Position"+str(i))
            self.labandata[positions[idx]] = self.getLabanotationKeyframeData(idx, time, dur, self.unfilteredLaban[i])
            idx = idx + 1


    #------------------------------------------------------------------------------
    # update energy markers and lines, and labanotation score
    #
    def updateEnergyPlotAndLabanScore(self, updateLabanScore=False):
        if self.ax is not None:
            if not self.points:
                return

            # Sort keyframe indices from the aggregated energy function.
            x = sorted(self.points.keys())

            # Plot aggregated energy keyframe markers:
            if not hasattr(self, 'line_ene_combined') or self.line_ene_combined is None:
                self.line_ene_combined, = self.ax.plot(
                    self.y_data['combined'], '*', color='k', mew=3, markersize=14, markevery=list(x)
                )
            else:
                self.line_ene_combined.set_data(range(len(self.y_data['combined'])), self.y_data['combined'])
                self.line_ene_combined.set_markevery(list(x))
                self.ax.draw_artist(self.line_ene_combined)

            # Plot vertical lines at keyframe locations:
            xs = np.array(list(x))
            lims = self.ax.get_ylim()
            # Create points for vertical lines for each keyframe index.
            x_points = np.repeat(xs[:, None], 3, axis=1).flatten()
            y_points = np.repeat(np.array(list(lims) + [np.nan])[None, :], len(xs), axis=0).flatten()
            if not hasattr(self, 'vlines') or self.vlines is None:
                self.vlines, = self.ax.plot(x_points, y_points, scaley=False, color='g')
            else:
                self.vlines.set_data(x_points, y_points)
                self.ax.draw_artist(self.vlines)

            self.ax.figure.canvas.draw_idle()

        # Optionally update Labanotation score if requested.
        if updateLabanScore and self.points:
            new_indices = sorted(self.points.keys())
            self.updateLaban(new_indices)
            settings.application.updateLaban(self.timeS, self.all_laban)

    #------------------------------------------------------------------------------
    #
    def add_point(self, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = int(x.xdata), int(x.ydata)

        y_on_curve = self.y_data[x]
        self.points[x] = y_on_curve

        return x, y_on_curve

    #------------------------------------------------------------------------------
    #
    def remove_point(self, x, _):
        if x in self.points:
            self.points.pop(x)

    #------------------------------------------------------------------------------
    #
    def setSelectedFrameMarker(self):
        if (self.ax is None):
            return

        cnt = len(self.jointFrames)
        idx = self.selectedFrame
        if ((idx is None) or (idx < 0) or (idx >= cnt)):
            return

        time = idx
        padding = 1.0 / 6.0

        if (self.selectedFrameMarker is None):
            yy = self.ax.get_ylim()
            self.selectedFrameMarker = patches.Rectangle((time-padding, yy[0]), 2*padding, (yy[1]-yy[0]), alpha=0.5, color='purple')
            self.ax.add_patch(self.selectedFrameMarker)
        else:
            self.selectedFrameMarker.set_x(time-padding)

    #------------------------------------------------------------------------------
    #
    def findNearestFrameForTime(self, time):
        cnt = len(self.jointFrames)
        if (cnt == 0):
            return None

        timeMS = time

        # find the frame corresponding to the given time
        for idx in range(0, cnt):
            kt = self.unfilteredTimeS[idx]

            if (kt == timeMS):
                return idx
            elif (kt > timeMS):
                break

        # should not get here if idx == 0, but let's check anyway
        if (idx == 0):
            return idx

        # now that we have an index, determine which frame time is closest to
        dist1 = abs(kt - time)
        dist2 = abs(self.unfilteredTimeS[idx-1] - time)

        return idx if (dist1 < dist2) else (idx-1)

    #------------------------------------------------------------------------------
    #
    def saveToJSON(self):
        filePath = settings.checkFileAlreadyExists(
            settings.application.outputFilePathJson, 
            fileExt=".json", 
            fileTypes=[('json files', '.json'), ('all files', '.*')]
        )
        if filePath is None:
            return

        file_name = os.path.splitext(os.path.basename(filePath))[0]

        labanjson = OrderedDict()
        labanjson[file_name] = self.labandata
    
        # Save the aggregated energy function to JSON.
        labanjson["energy_combined"] = list(self.y_data['combined'])

        try:
            with open(filePath, 'w') as file:
                json.dump(labanjson, file, indent=2)
                settings.application.logMessage(
                    f"Labanotation json script saved to '{settings.beautifyPath(filePath)}'"
                )
        except Exception as e:
            settings.application.logMessage(
                f"Exception saving Labanotation json script: {str(e)}"
            )

    #------------------------------------------------------------------------------
    #
    def saveToTXT(self):
        filePath = settings.checkFileAlreadyExists(settings.application.outputFilePathTxt, fileExt=".txt", fileTypes=[('text files', '.txt'), ('all files', '.*')])
        if (filePath is None):
            return

        # save text script
        script = settings.application.labanotation.labanToScript(self.timeS, self.all_laban)

        try:
            with open(filePath,'w') as file:
                file.write(script)
                file.close()
                settings.application.logMessage("Labanotation text script was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving Labanotation text script to '" + settings.beautifyPath(filePath) + "': " + str(e))

    #------------------------------------------------------------------------------
    #
    def selectTime(self, time):
        time = time * (self.duration)
        self.selectedFrame = self.findNearestFrameForTime(time)
        self.setSelectedFrameMarker()

    #------------------------------------------------------------------------------
    # find point closest to mouse position
    #
    def find_neighbor_point(self, event):
        distance_threshold = 3.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for x, y in self.points.items():
            distance = math.hypot(event.xdata - x, event.ydata - y) # euclidian norm
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        if min_distance < distance_threshold:
            return nearest_point
        return None

    # -----------------------------------------------------------------------------
    # canvas click event
    #
    def onCanvasClick(self, event):
        if (event.xdata is None) or (event.ydata is None):
            return

        # callback method for mouse click event
        # left click
        if event.button == 1 and event.inaxes in [self.ax]:
            if event.dblclick:
                pass
            else:
                self.dragging_sb = True

                # map xdata to [0..1]
                xx = self.ax.get_xlim()
                p = (event.xdata) / (xx[1]-xx[0])

                # call application so that other graphs can be updated as well
                settings.application.selectTime(p)

        # right click
        elif event.button == 3 and event.inaxes in [self.ax]:
            point = self.find_neighbor_point(event)
            if point and event.dblclick:
                self.remove_point(*point)
            elif point:
                self.dragging_point = point
                self.remove_point(*point)
            else:
                self.add_point(event)

            self.updateEnergyPlotAndLabanScore(True)

    # -----------------------------------------------------------------------------
    # canvas click release event
    #
    def onCanvasRelease(self, event):
        if event.button == 1 and event.inaxes in [self.ax] and self.dragging_sb:
            self.dragging_sb = False
        if event.button == 3 and event.inaxes in [self.ax] and self.dragging_point:
            self.add_point(event)
            self.dragging_point = None
            self.updateEnergyPlotAndLabanScore(True)

    # -----------------------------------------------------------------------------
    # canvas move event
    #
    def onCanvasMove(self, event):
        if (not self.dragging_sb or event.xdata is None) and (not self.dragging_point):
            return

        if self.dragging_sb:
            # map xdata to [0..1]
            xx = self.ax.get_xlim()
            p = event.xdata / (xx[1]-xx[0])

            # call application so that other graphs can be updated as well
            settings.application.selectTime(p)
        else:
            self.remove_point(*self.dragging_point)
            self.dragging_point = self.add_point(event)
            self.updateEnergyPlotAndLabanScore()

    #------------------------------------------------------------------------------
    #

