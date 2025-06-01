


import os
import torch
import numpy as np
from experta import *
from experta import utils
from . import physical_indices as pi
# Owlready2 ontology load/create
from owlready2 import get_ontology, sync_reasoner, Thing, DataProperty, ObjectProperty, FunctionalProperty
from numpy import array
#------------------------------------------------------------------------------
# Ontology load/create
#------------------------------------------------------------------------------
def load_or_create_ontology(path="laban.owl"):
    if os.path.exists(path):
        onto = get_ontology(f"file://{os.path.abspath(path)}").load()
    else:
        onto = get_ontology(f"file://{os.path.abspath(path)}")
        with onto:
#             class MotionSegment(Thing): pass
            class Symbol(Thing): pass
            class EffortQuality(Thing): pass
            class TimeQuality(Thing): pass
            class SpaceQuality(Thing): pass
            class FlowQuality(Thing): pass
            class Strong(EffortQuality): pass
            class Light(EffortQuality): pass
            class Sudden(TimeQuality): pass
            class Sustained(TimeQuality): pass
            class Direct(SpaceQuality): pass
            class Indirect(SpaceQuality): pass
            class BoundFlow(FlowQuality): pass
            class FreeFlow(FlowQuality): pass
            class jointName(DataProperty, FunctionalProperty): domain=[Symbol]; range=[str]
            class directionName(DataProperty, FunctionalProperty): domain=[Symbol]; range=[str]
            class levelName(DataProperty, FunctionalProperty): domain=[Symbol]; range=[str]
        onto.save(file=path)
    return onto

# onto = load_or_create_ontology()

#------------------------------------------------------------------------------
# Fact definition (includes kinematic deltas and foot contact)
#------------------------------------------------------------------------------
# print(type(iter([12,2,32])))
# ------------------------
# Fact Declarations
# ------------------------

class KeyframesLeft(Fact):
    """
    Maintains the list of remaining motion segments to process.
    - frames: Python list of CurrentMotionSegment data dicts
    """
    
    frames = Field(list, mandatory=True)

class FrameData(Fact):
    
    angles = Field(list, mandatory=True)
    lma_indices= Field(dict, mandatory=True)  
    relative_feet_heights=Field(dict, mandatory=True)  ,
    translation= Field(list, mandatory=True)  ,
    rotation= Field(list, mandatory=True)  

    
    
class CurrentMotionSegment(Fact): 
    """
    Raw frame data including spherical angles and global kinematics.
    Fields:
      frame_id: int
      phi: list of 12 floats
      theta: list of 12 floats
      translation: np.ndarray, shape (3,)
      rotation: float (yaw in degrees)
      footL_y: float
      footR_y: float
    """
    frame_id = Field(int, mandatory=True)
    phi = Field(list, mandatory=True)
    theta = Field(list, mandatory=True)
    lma_indices = Field(dict, mandatory=True)  
   
    translation = Field(list, mandatory=True,default=0.0)
    rotation = Field(float, mandatory=True, default=0.0)
    relative_feet_height = Field(list, mandatory=True)
    joints_left = Field(list, mandatory=True) 
  

class JointSphere(Fact):
    """
    Per-joint spherical data for classification.
    """
    frame_id = Field(int, mandatory=True)
    jointName = Field(str, mandatory=True)
    phi = Field(float, mandatory=True)
    theta = Field(float, mandatory=True)
    jointType = Field(str, mandatory=False, default="support")
class Direction(Fact):
    jointName = Field(str, mandatory=True)
    value     = Field(str, mandatory=True)
class Level    (Fact): 
    jointName=Field(str)
    value=Field(str)
class Support  (Fact): 
    frame_id=Field(int)
    
    base_translation =  Field(list, mandatory=True,default=0.0)
    base_rotation =  Field(float, mandatory=True,default=0.0)
    rot_support=Field(str)
    direction=Field(str)
    motion=Field(str)
    side=Field(str)

class Symbol(Fact):
    """
    Active Laban symbol for contiguous frames.
    """
    jointName = Field(str, mandatory=True)
    start = Field(int, mandatory=True)  
    
    duration = Field(int, mandatory=True)
    direction = Field(str, mandatory=True)
    level = Field(str, mandatory=False)

class Staff(Fact):
    """
    Final symbol lists per joint after all frames.
    """
    pass



# ------------------------
# Thresholds & params
# ------------------------

# ------------------------
# Rule-Based Engine
# ------------------------

class MotionClassifier(KnowledgeEngine):
    """
    Processes a list of frames one-by-one, classifying joint orientations
    into Laban symbols with variable thresholds per joint.

    Reset parameters:
      keyframes_left: list of dicts, each providing arguments for CurrentMotionSegment

    Workflow:
      1. Seed KeyframesLeft via DefFacts
      2. Pull next segment when available and all previous facts consumed
      3. Split into JointSphere facts
      4. TEST-rules classify directions and levels
      5. Lifecycle: create/extend/switch symbols
      6. When no frames left → aggregate Staff → halt
    """
    
    JOINT_NAMES =[
                    "elbowL", "wristL",
                    "shoulderL",
                    "ankleL", "footL", "kneeL",
                    "kneeR", "footR", "ankleR",
                    "shoulderR",
                    "wristR", "elbowR",
                    "head"
                ]
    TH = {jn:[0,22.5,67.5,112.5,157.5,180] for jn in JOINT_NAMES }
    STEP_TH = 0.05 ; JUMP_TH = 0.2; ROT_TH = 15

    @DefFacts()
    def _initialize(self):
        """Initialize with the provided keyframe list."""
        print("Initializing MotionClassifier")
        yield Fact(thresholds=self.TH, 
                   joint_names=self.JOINT_NAMES,
                   STEP_TH = 0.2, 
                   JUMP_TH = 0.1, 
                   ROT_TH = 15 , 
                   run=True)
    
    
    @Rule(
        # Pull next frame only when no pending segmentation/classification facts remain
        AS.kf << KeyframesLeft(frames=MATCH.frames & 
                               P(lambda frames: len(frames) > 0)),
        
        NOT(CurrentMotionSegment()),
        NOT(Support()),
        FrameData(  angles= MATCH.angles,
                    relative_feet_heights=MATCH.relative_feet_heights,
                    translation=MATCH.translation,
                    rotation=MATCH.rotation),
    )
    def initial_segment(self, kf, frames, angles, relative_feet_heights, translation, rotation):
        """
        Consume the head of the frames list to declare a CurrentMotionSegment,
        and update KeyframesLeft to the tail.
        """
        next_kf, *rest = frames
        print(f"\n Classification for frame {next_kf}")
      
        self.declare(CurrentMotionSegment( frame_id=next_kf,
                                            phi=[angles[next_kf][joint_id][1] for joint_id in range(13)],
                                            theta=[angles[next_kf][joint_id][0] for joint_id in range(13)],
                                            translation = translation[next_kf],
                                            rotation = rotation[next_kf],
                                            lma_indices={},
                                            relative_feet_height = relative_feet_heights[next_kf],#TODO,
                                            
                                            joints_left=self.JOINT_NAMES))
        self.declare(Support(frame_id=next_kf,
                             base_translation=translation[next_kf],
                             base_rotation=rotation[next_kf],
                             direction="None",
                             motion="None",
                             side="None",
                             rot_support="None"))
        self.modify(kf, frames=rest)
        self.declare(Staff({joint:[] for joint in self.JOINT_NAMES+["support_type","rotation"]}))


    
    @Rule(
        # Pull next frame only when no pending segmentation/classification facts remain#TODO: And when the support have been pushed.
        AS.kf << KeyframesLeft(frames=MATCH.frames & 
                               P(lambda frames: len(frames) > 0)),
        
        AS.seg <<CurrentMotionSegment(joints_left=MATCH.joints_left & 
                                      P(lambda joints_left: len(joints_left) == 0)),
        AS.sup << Support(frame_id=MATCH.frame_id,
                              direction= MATCH.direction & ~L('None'),
                             motion= MATCH.motion & ~L('None'),
                             side= MATCH.side & ~L('None'),
                             
                             rot_support = MATCH.rot_support & ~L("None")
            #              ),
                            # motion=Field(str)
                            # side=Field(str)
                        ),
        FrameData(  angles= MATCH.angles,
                    lma_indices=MATCH.lma,
                    relative_feet_heights=MATCH.relative_feet_heights,
                    translation=MATCH.translation,
                    
                    rotation=MATCH.rotation),
        NOT(JointSphere()),
        AS.staff<< Staff()
    )
    def next_segment(self, kf, seg, staff, sup,frames, angles, lma, relative_feet_heights,translation, rotation, direction, motion, side,rot_support ):
        """
        Consume the head of the frames list to declare a CurrentMotionSegment,
        and update KeyframesLeft to the tail.
        """
        next_kf, *rest = frames
        print(f"\n Classification for frame {next_kf}")
      
        self.retract(seg)
        self.declare(CurrentMotionSegment(  frame_id=next_kf,
                                            phi=[angles[next_kf][joint_id][1] for joint_id in range(13)],
                                            theta=[angles[next_kf][joint_id][0] for joint_id in range(13)], 
                                            translation = translation[next_kf],
                                            rotation = rotation[next_kf],
                                            lma_indices=lma[next_kf],
                                            relative_feet_height =relative_feet_heights[next_kf],#TODO,
                                            
                                            joints_left=self.JOINT_NAMES))
        
        
        self.modify(sup, frame_id=next_kf, direction="None", motion="None", side="None",
                             rot_support="None")
        self.modify(kf, frames=rest)
        
        self.retract(staff)
        staff_data=utils.unfreeze(dict(staff[0]))
        staff_data["support_type"].append([direction, motion, side])
        
        staff_data["rotation"].append([rot_support])
        self.declare(Staff(staff_data))


    @Rule(
        # Split each pulled segment into per-joint facts
       
        AS.seg <<CurrentMotionSegment(joints_left=MATCH.joints_left 
                                      & P(lambda joints_left: len(joints_left) > 0))
    )
    def declare_new_joint(self, seg, joints_left):
        
        # 1. remove the old fact

        new_joint, *rest = joints_left
        self.modify(seg, joints_left=rest)
        
        # 2. get all its data as a simple dict
        data = dict(seg)  
        self.declare(JointSphere(
            frame_id=data['frame_id'],
            jointName=new_joint,
            phi=data['phi'][self.JOINT_NAMES.index(new_joint)],
            theta=data['theta'][self.JOINT_NAMES.index(new_joint)],
            jointType= 'support' if new_joint.startswith('knee')
                               else 'body' 
                               if new_joint.startswith('torso')  else 'arm'
        ))
        
        # print(f"ID: {data['frame_id']} Declared new joint:{new_joint}")
    
    
    #When a joint is classified, create initial symbol, update existing one, or move to the next  
    @Rule( AS.js<< JointSphere(frame_id=MATCH.frame_id,
            jointName= MATCH.joint_name ),
           AS.direc << Direction(jointName=MATCH.joint_name,
                                 value=MATCH.direc_value), 
           
           AS.lvl << Level(jointName=MATCH.joint_name,
                                 value=MATCH.lvl_value), 
        #    AS.sup << Support(frame_id=MATCH.frame_id,
        #                      direction= ~L('None'),
        #                     #  motion= ~L('None'),
        #                     #  side= ~L('None'),
        #                  ),
           NOT(Symbol(jointName=MATCH.joint_name, 
                            start=MATCH.start),
                    TEST(lambda start,frame_id : start <= frame_id)),
           AS.staff<< Staff()
           ) 
           

    def create_initial_symbol(self,js, direc, staff, lvl, joint_name, frame_id, direc_value, 
                              lvl_value):
        self.retract(js)
        self.retract(direc)
        self.retract(lvl)
        #TODO: update staff and update symbol duration
        self.declare(Symbol(jointName=joint_name,
                            start=frame_id,
                            direction=direc_value, 
                            level=lvl_value,
                            duration=1))
        
        
        self.retract(staff)
        staff_data=utils.unfreeze(dict(staff[0]))
        staff_data[joint_name].append([direc_value,lvl_value])
        self.declare(Staff(staff_data))
        # print(self.facts)
        # print(f"ID: {frame_id} Classification of {joint_name} successful")
    
    
    #When a joint is classified, update the symbol or create a new one. 
    @Rule( AS.js<< JointSphere(frame_id=MATCH.frame_id,
            jointName= MATCH.joint_name ),
            AS.direc << Direction(jointName=MATCH.joint_name,
                                 value=MATCH.direc_value), 
            AS.lvl << Level(jointName=MATCH.joint_name,
                                 value=MATCH.lvl_value), 
            # AS.sup << Support(frame_id=MATCH.frame_id,
            #                  direction= ~L('None'),
            #                 #  motion= ~L('None'),
            #                 #  side= ~L('None'),
            #              ),
            AS.sym << Symbol(jointName=MATCH.joint_name, 
                            start=MATCH.start,
                            direction=MATCH.direc_value, 
                            level=MATCH.lvl_value),
                            
             TEST(lambda start,frame_id : start <= frame_id),
             AS.staff<< Staff())
    def update_symbol(self,js, sym, direc,lvl, start, direc_value, lvl_value, joint_name, frame_id,staff):
        self.retract(js)
        self.retract(direc)
        self.retract(lvl)
        
        self.modify(sym, duration=frame_id  -start)
        
        # update staff
        self.retract(staff)
        staff_data=utils.unfreeze(dict(staff[0]))
        staff_data[joint_name].append([direc_value,lvl_value])
        self.declare(Staff(staff_data))
        # print(f"ID: {frame_id} Classification of {joint_name} successful, updating symbol")
    
  
    
    @Rule( AS.js<< JointSphere(frame_id=MATCH.frame_id,
            jointName= MATCH.joint_name ),
            AS.direc << Direction(jointName=MATCH.joint_name,
                                 value=MATCH.direc_value),
            AS.lvl << Level(jointName=MATCH.joint_name,
                                 value=MATCH.lvl_value),  
            # AS.sup << Support(frame_id=MATCH.frame_id,
            #                  direction= ~L('None'),
            #                 #  motion= ~L('None'),
            #                 #  side= ~L('None'),
            #              ),
            AS.sym <<Symbol(jointName=MATCH.joint_name, 
                            start=MATCH.start,
                            direction= MATCH.sym_direction ,
                            level= MATCH.sym_level
                            ),
            TEST(lambda start,frame_id : start <= frame_id),
            (TEST(lambda sym_direction, direc_value:sym_direction != direc_value) |
            TEST(lambda sym_level, lvl_value: sym_level!= lvl_value)),
            AS.staff<< Staff())
    def move_to_next_symbol(self,js, direc,lvl, sym, joint_name, frame_id, direc_value,lvl_value, staff):
        self.retract(js)
        self.retract(direc)
        self.retract(lvl)
        self.retract(sym)
        
        self.declare(Symbol(jointName=joint_name,
                            start=frame_id,
                            direction=direc_value, 
                            
                            level= lvl_value,
                            duration=1))
        
         # update staff
        self.retract(staff)
        staff_data=utils.unfreeze(dict(staff[0]))
        staff_data[joint_name].append([direc_value,lvl_value])
        self.declare(Staff(staff_data))
        
        # print(f"ID: {frame_id} Classification of {joint_name} successful, creating symbol")
    
    @Rule(
            AS.sym << Symbol(jointName=MATCH.joint_name),
            NOT(JointSphere()),
            NOT(CurrentMotionSegment()),
            KeyframesLeft(frames=P(lambda frames: len(frames) == 0)),
            AS.staff<< Staff()
                            )
    def finish_symbols(self, sym, joint_name, staff):
        
        self.retract(sym)
         # update staff
         
        # self.retract(staff)
        # sym_data = dict(sym)
        # staff_data=utils.unfreeze(dict(staff[0]))
        # staff_data[joint_name].append([sym_data["start"],sym_data['direction'],sym_data['level']])
    
        # self.declare(Staff(staff_data))
        # print(f"END:Pushing last symbol {joint_name} to staff")
    
    @Rule(
        # When frames list is empty and no pending facts, signal end
        # Pull next frame only when no pending segmentation/classification facts remain
        KeyframesLeft(frames=P(lambda frames: len(frames) == 0)),
        
        AS.seg<< CurrentMotionSegment(joints_left=P(lambda joints_left: len(joints_left) == 0)),
        AS.sup << Support(frame_id=MATCH.frame_id,
                              direction= MATCH.direction & ~L('None'),
                            motion= MATCH.motion & ~L('None'),
                             side= MATCH.side & ~L('None'),
                             rot_support = MATCH.rot_support & ~L("None")
            #              ),
            #              ),
                            # motion=Field(str)
                            # side=Field(str)
                        ),
        AS.staff<< Staff()
    )
    def delete_last_motion_seg(self,seg, staff, direction, motion,side, rot_support):
      
        print("\n Last keframe processed, ending classification.")
        # Remove the last segment fact
        self.retract(seg)
        
        self.retract(staff)
        staff_data=utils.unfreeze(dict(staff[0]))
        staff_data["support_type"].append([direction,motion,side])
        
        staff_data["rotation"].append([rot_support])
        self.declare(Staff(staff_data))
      
    
    # When no joints left, check if there are any pending frames
    @Rule(
        # When frames list is empty and no pending facts, signal end
         # Pull next frame only when no pending segmentation/classification facts remain
        AS.kf << KeyframesLeft(frames=P(lambda frames: len(frames) == 0)),
        
        NOT(CurrentMotionSegment()),
        AS.data << FrameData(), 
        NOT(JointSphere()),
        NOT(Symbol())
    )
    def end_classification(self,data):
        print("No keyframes left, ending classification.")
        self.retract(data)
        self.declare(Fact(end=True))
        
    
    # # ---------- Direction Classification Rules ----------
        
    # — Direction Classification (8 compass regions) —
    @Rule(
        JointSphere(phi=P(lambda ph: (0 <= ph <= 22.5) or (-22.5 < ph < 0)),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_forward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Forward'))

    @Rule(
        JointSphere(phi=P(lambda ph: 22.5 < ph <= 67.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_left_forward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Left Forward'))

    @Rule(
        JointSphere(phi=P(lambda ph: 67.5 < ph <= 112.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_left(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Left'))

    @Rule(
        JointSphere(phi=P(lambda ph: 112.5 < ph <= 157.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_left_backward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Left Backward'))

    @Rule(
        JointSphere(phi=P(lambda ph: (ph > 157.5 and ph <= 180) or (ph <= -157.5 and ph > -180)),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_backward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Backward'))

    @Rule(
        JointSphere(phi=P(lambda ph: -157.5 < ph <= -112.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_right_backward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Right Backward'))

    @Rule(
        JointSphere(phi=P(lambda ph: -112.5 < ph <= -67.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_right(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Right'))

    @Rule(
        JointSphere(phi=P(lambda ph: -67.5 < ph <= -22.5),
                    jointName=MATCH.joint_name),
        NOT(Direction(jointName=MATCH.joint_name))
    )
    def dir_right_forward(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Right Forward'))




    # — Level Classification —
    # Generic for non-support & non-torso
    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: th < 22.5)),
        NOT(Level(jointName=MATCH.joint_name)),
        NOT(Direction(jointName=MATCH.joint_name))
        
    )
    def level_place_high_new(self, joint_name):

        self.declare(Direction(jointName=joint_name, value='Place'))
        self.declare(Level(jointName=joint_name, value='High'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: th < 22.5)),
        NOT(Level(jointName=MATCH.joint_name)),
        AS.dir<<Direction(jointName=MATCH.joint_name)
        
    )
    def level_place_high_mod(self, joint_name,dir):
   
        self.declare(Level(jointName=joint_name, value='High'))
        self.modify(dir, value='Place')


    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: 22.5 <= th < 67.5)),
        NOT(Level(jointName=MATCH.joint_name))
    )
    def level_high(self, joint_name):
        self.declare(Level(jointName=joint_name, value='High'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'),# & ~L('torso'),
                    theta=P(lambda th: 67.5 <= th < 112.5)),
        NOT(Level(jointName=MATCH.joint_name))
    )
    def level_normal(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Normal'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: 112.5 <= th < 157.5)),
        NOT(Level(jointName=MATCH.joint_name))
    )
    def level_low(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Low'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: th >= 157.5)),
        NOT(Level(jointName=MATCH.joint_name),

        NOT(Direction(jointName=MATCH.joint_name)))
    )
    def level_place_low_new(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Place'))
        self.declare(Level(jointName=joint_name, value='Low'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support'), #& ~L('torso'),
                    theta=P(lambda th: th >= 157.5)),
        NOT(Level(jointName=MATCH.joint_name)),

        AS.dir<< Direction(jointName=MATCH.joint_name)
    )
    def level_place_low_mod(self, joint_name,dir):
        self.modify(dir, value='Place')
        self.declare(Level(jointName=joint_name, value='Low'))
        
    @Rule(
    JointSphere(jointName=MATCH.joint_name,
                jointType=MATCH.jt & ~L('support') & ~L('body'),
                theta=P(lambda th: th < 15)),
    NOT(Level(jointName=MATCH.joint_name)),
    NOT(Direction(jointName=MATCH.joint_name))
    )
    def level_place_high_nonbody_new(self, joint_name):
        self.declare(Direction(jointName=joint_name, value='Place'))
        self.declare(Level(jointName=joint_name, value='High'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & ~L('support')  & ~L('body'),
                    theta=P(lambda th: 15 <= th < 30)),
        NOT(Level(jointName=MATCH.joint_name))
    )
    def level_high_nonbody(self, joint_name):
        self.declare(Level(jointName=joint_name, value='High'))

# …add the rest of the non-body bands (30–67.5 → Normal, 67.5–112.5 → Low, ≥112.5 → Place Low)… 

    # Support-level bands


    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & L('support'), #& & ~L('body'),
                    theta=P(lambda th: th < 67.5)),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: (joint_name[-1]!=side[0] or motion=="Jump" )))
    def strict_high(self, joint_name):
        self.declare(Level(jointName=joint_name, value='High'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & L('support'), #&  & ~L('body'),
                    theta=P(lambda th: 67.5<= th < 112.5)),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: (joint_name[-1]!=side[0] or motion=="Jump" ))
    )
    def strict_normal(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Normal'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=MATCH.jt & L('support'),# & ~L('body'),
                    theta=P(lambda th:  112.5 <= th )),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: (joint_name[-1]!=side[0] or motion=="Jump" ))
    )
    def strict_low(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Low'))



    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=L('support'),
                    theta=P(lambda th: th < 90)),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: 
            (joint_name[-1]==side[0] or side=="Both")
            and (motion=="Stand" or motion=="Squat")))
    def support_low_o(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Low o'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=L('support'),
                    theta=P(lambda th: 90 <= th < 120)),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: 
            (joint_name[-1]==side[0] or side=="Both")
            and (motion=="Stand"))
        )
    
    def support_normal_o(self, joint_name):
        self.declare(Level(jointName=joint_name, value='Normal o'))

    @Rule(
        JointSphere(jointName=MATCH.joint_name,
                    jointType=L('support'),
                    theta=P(lambda th: th >= 120)),
        NOT(Level(jointName=MATCH.joint_name)),
        Support( motion= MATCH.motion,
                 side = MATCH.side ),
        TEST(lambda motion,side,joint_name: 
            (joint_name[-1]==side[0] or side=="Both")
            and (motion=="Stand"))
        )
    
    def support_high_o(self, joint_name):
        self.declare(Level(jointName=joint_name, value='High o'))
    
    
    
    
    # # ---------- Global translation direction ----------
    
    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             (translation[0] - base_translation[0]) > STEP_TH/2 and
             (translation[2] - base_translation[2]) > STEP_TH/2)
    )
    def support_right_forward(self, sup, base_translation, STEP_TH):
        
        new_base_translation=utils.unfreeze(base_translation)
        # Move both axes positively
        new_base_translation[0] += STEP_TH
        new_base_translation[2] += STEP_TH
        self.modify(sup,
                    direction='Right Forward',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             (translation[0] - base_translation[0]) < -STEP_TH/2 and
             (translation[2] - base_translation[2]) > STEP_TH/2)
    )
    def support_left_forward(self, sup, base_translation, STEP_TH):
        
        new_base_translation=utils.unfreeze(base_translation)
        # Depth positive, side negative
        new_base_translation[0] -= STEP_TH
        new_base_translation[2] += STEP_TH
        self.modify(sup,
                    direction='Left Forward',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             abs(translation[0] - base_translation[0]) <= STEP_TH/2 and
             translation[2] - base_translation[2] > STEP_TH/2)
    )
    def support_forward(self, sup, base_translation, STEP_TH):
        
        new_base_translation=utils.unfreeze(base_translation)
        # Depth positive, minimal side
        new_base_translation[2] += STEP_TH
        self.modify(sup,
                    direction='Forward',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             (translation[0] - base_translation[0]) > STEP_TH/2 and
             (translation[2] - base_translation[2]) < -STEP_TH/2 )
    )
    def support_right_backward(self, sup, base_translation, STEP_TH):
        new_base_translation=utils.unfreeze(base_translation)
        # Depth negative, side positive
        new_base_translation[0] += STEP_TH
        new_base_translation[2] -= STEP_TH
        self.modify(sup,
                    direction='Right Backward',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             (translation[0] - base_translation[0]) < -STEP_TH/2 and
             (translation[2] - base_translation[2]) < -STEP_TH/2)
    )
    def support_left_backward(self, sup, base_translation, STEP_TH):
        new_base_translation=utils.unfreeze(base_translation)
        # Both axes negative
        new_base_translation[0] -= STEP_TH
        new_base_translation[2] -= STEP_TH
        self.modify(sup,
                    direction='Left Backward',
                    base_translation=new_base_translation)
    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             abs(translation[0] - base_translation[0]) <= STEP_TH/2 and
             (translation[2] - base_translation[2]) < -STEP_TH/2)
    )
    def support_backward(self, sup, base_translation, STEP_TH):
        new_base_translation=utils.unfreeze(base_translation)
        # Side positive, minimal depth
        new_base_translation[2] -= STEP_TH
        self.modify(sup,
                    direction='Backward',
                    base_translation=new_base_translation)
        
    @Rule( 
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH: (
             ((translation[0] - base_translation[0]) > STEP_TH/2) and
             (abs(translation[2] - base_translation[2]) <= STEP_TH/2)) )
    )
    def support_right(self, sup, base_translation, STEP_TH):
        new_base_translation=utils.unfreeze(base_translation)
        # Side positive, minimal depth
        new_base_translation[0] += STEP_TH
        self.modify(sup,
                    direction='Right',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
            (translation[0] - base_translation[0]) < -STEP_TH/2 and
            abs(translation[2] - base_translation[2]) <= STEP_TH/2))
    
    def support_left(self, sup, base_translation, STEP_TH):
        new_base_translation=utils.unfreeze(base_translation)
        # Side negative, minimal depth
        new_base_translation[0] -= STEP_TH
        self.modify(sup,
                    direction='Left',
                    base_translation=new_base_translation)

    @Rule(
        Fact(STEP_TH=MATCH.STEP_TH),
        CurrentMotionSegment(frame_id=MATCH.frame_id,
                             translation=MATCH.translation,
                             rotation=MATCH.rotation),
        AS.sup << Support(frame_id=MATCH.frame_id,
                          direction=L('None'),
                          base_translation=MATCH.base_translation,
                          base_rotation=MATCH.base_rotation),
        TEST(lambda base_translation, translation, STEP_TH:
             abs(translation[0] - base_translation[0]) <= STEP_TH/2 and
             abs(translation[2] - base_translation[2]) <= STEP_TH/2)
    )
    def support_place(self, sup):
        # Minimal movement
        self.modify(sup,
                    direction='Place')
    
    # # ---------- Global translation type ----------
    
    @Rule(
        Fact(JUMP_TH=MATCH.JUMP_TH),
        CurrentMotionSegment(
            frame_id    = MATCH.frame_id,
            translation = MATCH.translation,
            rotation    = MATCH.rotation,
            relative_feet_height = MATCH.relative_feet_height
        ),
        AS.sup << Support(
            frame_id          = MATCH.frame_id,
            motion            = L('None'),
            side              = L('None'),
            base_translation  = MATCH.base_translation,
            base_rotation     = MATCH.base_rotation
        ),
        TEST(lambda relative_feet_height, JUMP_TH:
             (abs(relative_feet_height[0] ) > JUMP_TH/2) and
             (abs( relative_feet_height[1] ) > JUMP_TH/2)
        )
    )
    def jump_both(self, sup, base_translation, JUMP_TH):
        new_base_translation=utils.unfreeze(base_translation)

        # Both feet airborne: increment vertical component
        new_base_translation[1] += JUMP_TH
        self.modify(sup,
                    motion='Jump',
                    side='Both',
                    base_translation=new_base_translation)

    # 2) Squat when at least one foot is on (or near) ground, but vertical move < -JUMP_TH
    @Rule(
        Fact(JUMP_TH=MATCH.JUMP_TH),
        CurrentMotionSegment(
            frame_id      = MATCH.frame_id,
            translation   = MATCH.translation,
            rotation      = MATCH.rotation,
            relative_feet_height = MATCH.relative_feet_height
        ),
        AS.sup << Support(
            frame_id          = MATCH.frame_id,
            motion            = L('None'),
            side              = L('None'),
            base_translation  = MATCH.base_translation,
            base_rotation     = MATCH.base_rotation
        ),
        TEST(lambda relative_feet_height, translation, base_translation, JUMP_TH:
             # At least one foot “touching” ground (<= JUMP_TH/2)
             ((abs(relative_feet_height[0] ) <= JUMP_TH/2) or
              (abs(relative_feet_height[1] ) <= JUMP_TH/2))
             # And vertical move downward beyond threshold
             and (translation[1] - base_translation[1] < -JUMP_TH)
        )
    )
    def squat_both(self, sup, base_translation):
        # Foot contact & moving downward: squat, both feet supporting
        # “Squat” does not change base_translation[1] except to reset to 0
        new_base_translation=utils.unfreeze(base_translation)
        new_base_translation[1] = 0.0
        self.modify(sup,
                    motion='Squat',
                    side='Both',
                    base_translation=new_base_translation)

    # 3) Support LEFT when left foot on ground and right foot off
    @Rule(
        Fact(JUMP_TH=MATCH.JUMP_TH),
        CurrentMotionSegment(
            frame_id      = MATCH.frame_id,
            translation   = MATCH.translation,
            rotation      = MATCH.rotation,
            relative_feet_height = MATCH.relative_feet_height
        ),
        AS.sup << Support(
            frame_id          = MATCH.frame_id,
            motion            = L('None'),
            side              = L('None'),
            base_translation  = MATCH.base_translation,
            base_rotation     = MATCH.base_rotation
        ),
        TEST(lambda  relative_feet_height, JUMP_TH:
             # Left foot on/near ground AND right foot off ground
             (abs(relative_feet_height[0]) <= JUMP_TH/2) and
             (abs(relative_feet_height[1] ) >  JUMP_TH/2)
        )
    )
    def right_step(self, sup):
        # Left foot supporting only
        self.modify(sup,
                    motion='Stand',
                    side='Left')

    # 4) Support RIGHT when right foot on ground and left foot off
    @Rule(
        Fact(JUMP_TH=MATCH.JUMP_TH),
        CurrentMotionSegment(
            frame_id      = MATCH.frame_id,
            translation   = MATCH.translation,
            rotation      = MATCH.rotation,
            relative_feet_height = MATCH.relative_feet_height
        ),
        AS.sup << Support(
            frame_id          = MATCH.frame_id,
            motion            = L('None'),
            side              = L('None'),
            base_translation  = MATCH.base_translation,
            base_rotation     = MATCH.base_rotation
        ),
        TEST(lambda relative_feet_height, JUMP_TH:
             # Right foot on/near ground AND left foot off ground
             (abs(relative_feet_height[0] ) >  JUMP_TH/2) and
             (abs(relative_feet_height[1]) <= JUMP_TH/2)
        )
    )
    def step_left(self, sup):
        # Right foot supporting only
        self.modify(sup,
                    motion='Stand',
                    side='Right')

    # 5) Support BOTH feet (standing) when neither jump nor squat and both feet are on/near ground
    @Rule(
        Fact(JUMP_TH=MATCH.JUMP_TH),
        CurrentMotionSegment(
            frame_id      = MATCH.frame_id,
            translation   = MATCH.translation,
            rotation      = MATCH.rotation,
            relative_feet_height = MATCH.relative_feet_height
        ),
        AS.sup << Support(
            frame_id          = MATCH.frame_id,
            motion            = L('None'),
            side              = L('None'),
            base_translation  = MATCH.base_translation,
            base_rotation     = MATCH.base_rotation
        ),
        TEST(lambda relative_feet_height, JUMP_TH, translation, base_translation:
             # Both feet on/near ground
             (abs( relative_feet_height[0]) <= JUMP_TH/2) and
             (abs( relative_feet_height[1]) <= JUMP_TH/2)
             # and no downward squat (translation[1] - base_translation[1] >= -JUMP_TH)
             and (translation[1] - base_translation[1] >= -JUMP_TH)
        )
    )
    def stand(self, sup):
        # Both feet supporting (standing)
        self.modify(sup,
                    motion='Stand',
                    side='Both')

    @Rule(
        Fact(ROT_TH=MATCH.ROT_TH),
        CurrentMotionSegment(
            frame_id         = MATCH.frame_id,
            rotation = MATCH.part_rot
        ),
        AS.sup << Support(
            frame_id       = MATCH.frame_id,
            base_rotation  = MATCH.base_rot,
            
            rot_support = L("None"),
        )
        
        )
    def update_rotation(self, sup, part_rot, base_rot, ROT_TH):
        # 1) Compute raw delta and wrap it exactly as in your procedural code
        delta_raw   = part_rot - base_rot
        delta_deg   = np.rad2deg(delta_raw)
        delta_yaw_deg = ((delta_deg + 180) % 360) - 180

        rotation = np.sign(delta_yaw_deg) * ROT_TH*(abs(delta_yaw_deg)//ROT_TH)
        
        # 3) Update base_rotation[2] by adding rot_step_rad, then wrap to [-π, π]
        new_base = utils.unfreeze(base_rot)
        new_base_yaw = new_base + np.deg2rad(rotation)
        # Wrap into [-pi, +pi]
        new_base = ((new_base_yaw + np.pi) % (2 * np.pi)) - np.pi

        # 4) Modify the Support fact’s base_rotation in working memory
        self.modify(sup, base_rotation=new_base, rot_support=str(rotation))
    
def run_classification(joints_info, keyframes, fps=120):
    """
    frames_list: list of dicts with keys: frame_id, phi, theta,
      translation(np.array), rotation(float), footL_y, footR_y
    Returns: staff_data: [ [ (direction,level) per joint... ], translation, rotation, (direction,motion,side) ]
    """
    joints_vector, partial_translation, partial_rotation=joints_info
    keyframes=[0]+keyframes+[joints_vector.shape[0]-1]
    engine = MotionClassifier()
    engine.reset()
    times=[int(1+ i*1000/fps) for i in range(len(joints_vector))]
    spherical, lma, relative_feet_heights=pi.calculate_physical_indices(joints_vector, keyframes, times, 120)
    engine.declare(KeyframesLeft(frames=keyframes))
    
    
    engine.declare(FrameData(
        angles=spherical.tolist(),
        lma_indices=dict(zip(keyframes[1:],lma)),
        relative_feet_heights=dict(zip(keyframes,relative_feet_heights)),
        
        translation= partial_translation.tolist(),
        rotation= partial_rotation[:,1].tolist()
    ))
    # engine.declare(Fact(end=True))
    
    engine.run()
    
    # print(engine.facts)
    print("done")
    staff_list=[]
    staff = []
    for fact in engine.facts.values():
        
        if isinstance(fact, Staff):
            staff = utils.unfreeze(dict(fact)[0])
            break
    
    # collect per-joint latest symbols
    staff_list=np.array(list(staff.values())).T.tolist()
    
    # staff.supports = [(sup['direction'], sup['motion'], sup['side']) for sup in engine.facts.values() if isinstance(sup, SupportSymbol)]
    return staff_list

# import random
# run_classification([1,4,5],[[[(360.0*random.random()-180.0,180.0*random.random() )for _ in range(12)] for _ in range(6)], 
#                           [[1,2,3] for i in range(6)],
#                           [1.0  for i in range(6)],
#                           [1.0 for i in range(6)],
#                           [1.0  for i in range(6)]])


# class SegmentFact(Fact):
    
    
#     n = Field(lambda n: isinstance(n, int) and n >= 0, mandatory=True)
#     result = Field(int, mandatory=True)
    
    
#     segment    = Field(object)
#     angles     = Field(dict)       # {joint: (theta,phi)}
   
#     partial_translation   = Field(float)      # horizontal translation norm
#     delta_y    = Field(float)      # vertical translation delta
#     delta_yaw  = Field(float)      # yaw change in degrees
#     foot_L     = Field(float)      # left foot height
#     foot_R     = Field(float)      # right foot height
#     weight     = Field(float)
#     timeq      = Field(float)
#     spaceq     = Field(float)
#     flowq      = Field(float)

# #------------------------------------------------------------------------------
# # Ontology classes references
# #------------------------------------------------------------------------------
# MotionSegment = onto.MotionSegment
# Symbol        = onto.Symbol
# Strong        = onto.Strong
# Light         = onto.Light
# Sudden        = onto.Sudden
# Sustained     = onto.Sustained
# Direct        = onto.Direct
# Indirect      = onto.Indirect
# BoundFlow     = onto.BoundFlow
# FreeFlow      = onto.FreeFlow

# #------------------------------------------------------------------------------
# # Module 3: Spherical Angles & LMA Indices
# #------------------------------------------------------------------------------
# # parent map
# PARENT_REL = {
#     "elbowR":"shoulderR","elbowL":"shoulderL",
#     "wristR":"elbowR","wristL":"elbowL",
#     "kneeR":"hipR","kneeL":"hipL",
#     "ankleR":"kneeR","ankleL":"kneeL",
#     "footR":"ankleR","footL":"ankleL",
#     "head":"neck","spineS":"spineM"
# }
# AMASS_TO_KINECT_MAP = {"spineB":0,"spineM":3,"spineS":6,"neck":12,"head":15,
#     "shoulderL":16,"elbowL":18,"wristL":20,"handL":25,
#     "shoulderR":17,"elbowR":19,"wristR":21,"handR":41,
#     "hipL":1,"kneeL":4,"ankleL":7,"footL":10,
#     "hipR":2,"kneeR":5,"ankleR":8,"footR":11,
#     "handTL":34,"thumbL":35,"handTR":49,"thumbR":50}


# WEIGHTS = {AMASS_TO_KINECT_MAP[k]:w for k,w in {
#     'spineB':0.497,'shoulderL':0.028,'shoulderR':0.028,
#     'elbowL':0.016,'elbowR':0.016,'handL':0.006,'handR':0.006,
#     'hipL':0.10,'hipR':0.10,'kneeL':0.0465,'kneeR':0.0465,
#     'footL':0.0145,'footR':0.0145,'head':0.081}.items()}


# class MotionSeg(Fact):
    
    
#     n = Field(lambda n: isinstance(n, int) and n >= 0, mandatory=True)
#     result = Field(int, mandatory=True)
    
    
    
# class ComputeFactorial(KnowledgeEngine):
#     @DefFacts()
#     def first(self):
#         yield Factorial(n=0, result=1)

#     @Rule(
#         AS.f << Factorial(
#             n=MATCH.n,
#             result=MATCH.r))
#     def factorial(self, f, n, r):
#         self.declare(
#             Factorial(
#                 n=n + 1,
#                 result=(n + 1) * r))
#         self.retract(f)

# def to_sphere(vec):
#     r = np.linalg.norm(vec)
#     if r<1e-8: return 0,0,0
#     theta = np.degrees(np.arccos(vec[2]/r))
#     phi   = np.degrees(np.arctan2(vec[1],vec[0]))
#     return r,theta,phi

# def calculate_base_rotation(joint):
#     shL = np.zeros(3)
#     shR = np.zeros(3)
#     spM = np.zeros(3)

#     shL[0] = joint[0]['shoulderL']['x']
#     shL[1] = joint[0]['shoulderL']['y']
#     shL[2] = joint[0]['shoulderL']['z']
#     shR[0] = joint[0]['shoulderR']['x']
#     shR[1] = joint[0]['shoulderR']['y']
#     shR[2] = joint[0]['shoulderR']['z']

#     spM[0] = joint[0]['spineM']['x']
#     spM[1] = joint[0]['spineM']['y']
#     spM[2] = joint[0]['spineM']['z']

#     # convert kinect space to spherical coordinate
#     # 1. normal vector of plane defined by shoulderR, shoulderL and spineM
#     sh = np.zeros((3,3))
#     v1 = shL-shR
#     v2 = [0,-1,0]# spM-shR
#     sh[0] = np.cross(v2,v1)#x axis
#     sh[1] = v1#y axis
#     sh[2] = np.cross(sh[0],sh[1])#z axis
#     nv = np.zeros((3,3))
#     nv[0] = norm1d(sh[0])
#     nv[1] = norm1d(sh[1])
#     nv[2] = norm1d(sh[2])
#     # 2. generate the rotation matrix for
#     # converting point from kinect space to euculid space, then sphereical
#     base_rotation = np.transpose(nv)
#     return base_rotation

# def raw2sphere(jf,joint, base_rot=None, base_trans=None):
#     positions = {joint: np.array([jf[0][joint]['x'][0],jf[0][joint]['y'][0],jf[0][joint]['z'][0]])
#                  for joint in jf[0].dtype.names if joint in AMASS_TO_KINECT_MAP}
#     conv = calculate_base_rotation(joint)
#     coords = []
#     for child,parent in PARENT_REL.items():
#         vec = positions[child] - positions[parent]
#         coords.append(to_sphere(conv.T.dot(vec)))
#     return coords

# def compute_lma_indices(jp, keyframes, fps=120):
#     dt=1.0/fps
    
#     segs=[]
    
#     frames=[0]+[kf+1 for kf in keyframes]+[jp.shape[0]-1]
    
#     for i in range(len(frames)-1):
#         s,e=frames[i],frames[i+1]
#         seg=jp[s:e+1];M=seg.shape[0]
#         if M<5:
#             segs.append({'weight':0,'time':0,'space':0,'flow':0});continue
#         vel = (seg[2:]-seg[:-2])/(2*dt); acc=(seg[2:]-2*seg[1:-1]+seg[:-2])/(dt*dt)
#         jerk=(seg[4:]-2*seg[3:-1]+2*seg[1:-3]-seg[:-4])/(2*dt**3)
#         vn,an,jn = np.linalg.norm(vel,2,2),np.linalg.norm(acc,2,2),np.linalg.norm(jerk,2,2)
#         # weight
#         E=np.zeros(vn.shape[0])
#         for k,a in WEIGHTS.items(): E+=a*(vn[:,k]**2)
#         w=E.max()
#         # time
#         T=(an.mean(0)*list(WEIGHTS.values())).sum()
#         # space
#         disp=np.linalg.norm(seg[1:]-seg[:-1],2,2);
#         net=np.linalg.norm(seg[-1]-seg[0],2,1)
#         S=0
#         for k,a in WEIGHTS.items(): S+=a*(disp[:,k].sum()/(net[k] if net[k]>1e-6 else 1e-6))
#         # flow
#         F=(jn.mean(0)*list(WEIGHTS.values())).sum()
#         segs.append({'weight':w,'time':T,'space':S,'flow':F})
#     return segs



# class LabanEngine(KnowledgeEngine):
#     @Rule(SegmentFact(angles=MATCH.ang, segment=MATCH.seg))
#     def make_symbols(self, ang, seg):
#         syms=[]
#         for j,(th,ph) in ang.items():
#             if -22.5<=ph<=22.5: d='Forward'
#             elif 22.5<ph<=67.5: d='LeftForward'
#             elif 67.5<ph<=112.5: d='Left'
#             elif 112.5<ph<=157.5: d='LeftBackward'
#             elif ph>157.5 or ph<-157.5: d='Backward'
#             else: d='Right'
#             if th<22.5: l='High'
#             elif th<67.5: l='Normal'
#             elif th<112.5: l='Low'
#             else: l='PlaceLow'
#             s=Symbol(); s.jointName=[j]; s.directionName=[d]; s.levelName=[l]
#             syms.append(s)
#         seg.hasSymbol = syms
#     @Rule(SegmentFact(foot_L=P(lambda f:f>0.1), foot_R=P(lambda f:f>0.1), segment=MATCH.seg))
#     def jump(self, seg): seg.hasSpaceQuality=[Direct()]
#     @Rule(SegmentFact(foot_L=P(lambda f:f<=0.1), foot_R=P(lambda f:f<=0.1), delta_y=P(lambda dy: dy< -0.1), segment=MATCH.seg))
#     def squat(self, seg): seg.hasSpaceQuality=[Indirect()]
#     @Rule(SegmentFact(delta_xz=P(lambda dx: dx>0.01), segment=MATCH.seg))
#     def step(self, seg): seg.hasSpaceQuality=[Direct()]
#     @Rule(SegmentFact(delta_xz=P(lambda dx: dx<=0.01), segment=MATCH.seg))
#     def stand(self, seg): seg.hasSpaceQuality=[Indirect()]

#     # Turn detection rules
#     @Rule(SegmentFact(delta_yaw=P(lambda y:y>15), segment=MATCH.seg))
#     def turn_left(self, seg): seg.hasFlowQuality=[BoundFlow()]
#     @Rule(SegmentFact(delta_yaw=P(lambda y:y< -15), segment=MATCH.seg))
#     def turn_right(self, seg): seg.hasFlowQuality=[BoundFlow()]
#     @Rule(SegmentFact(delta_yaw=P(lambda y:-15<=y<=15), segment=MATCH.seg))
#     def no_turn(self, seg): seg.hasFlowQuality=[FreeFlow()]

#     # LMA Effort: Weight
#     @Rule(SegmentFact(weight=P(lambda w:w>1.0), segment=MATCH.seg))
#     def weight_strong(self, seg): seg.hasEffortQuality=[Strong()]
#     @Rule(SegmentFact(weight=P(lambda w:w<=1.0), segment=MATCH.seg))
#     def weight_light(self, seg): seg.hasEffortQuality=[Light()]

#     # Time Effort
#     @Rule(SegmentFact(timeq=P(lambda t:t>1.0), segment=MATCH.seg))
#     def time_sudden(self, seg): seg.hasTimeQuality=[Sudden()]
#     @Rule(SegmentFact(timeq=P(lambda t:t<=1.0), segment=MATCH.seg))
#     def time_sustained(self, seg): seg.hasTimeQuality=[Sustained()]

#     # Flow Effort (already partly used by turn)
#     @Rule(SegmentFact(flowq=P(lambda f:f>1.0), segment=MATCH.seg))
#     def flow_bound(self, seg): seg.hasFlowQuality=[BoundFlow()]
#     @Rule(SegmentFact(flowq=P(lambda f:f<=1.0), segment=MATCH.seg))
#     def flow_free(self, seg): seg.hasFlowQuality=[FreeFlow()]
    
#     @Rule(SegmentFact(spaceq=P(lambda s:s>1.0),segment=MATCH.seg))
#     def direct(self,seg): seg.hasSpaceQuality=[onto.Direct()]
#     @Rule(SegmentFact(spaceq=P(lambda s:s<=1.0),segment=MATCH.seg))
#     def indirect(self,seg): seg.hasSpaceQuality=[onto.Indirect()]
   

# # pipeline

# def run_KBS(joints_info, keyframes, fps=120):
    
#     joints, jointsFrames, base_translation, base_rotation = joints_info
#     sph=raw2sphere(joints, jointsFrames,keyframes)
#     # LMA indices
#     lma=compute_lma_indices(joints,keyframes)
        
#     engine=LabanEngine(); engine.reset()
#     segs=[]
#     for idx,kf in enumerate(keyframes):
        
#         seg=onto.MotionSegment(f"Seg{idx}")
#         # compute deltas
#         prev = kf-1 if kf>0 else kf
#         dxz = np.linalg.norm(base_translation[kf,[0,2]] - base_translation[prev,[0,2]])
#         dy  = translations[kf,1] - translations[prev,1]
#         dyaw= np.degrees(rotations[kf,2] - rotations[prev,2])
#         # declare
#         engine.declare(SegmentFact(
#             segment   = seg,
#             angles    = joints[kf],
#             delta_xz  = float(dxz),
#             delta_y   = float(dy),
#             delta_yaw = float(dyaw),
#             foot_L    = float(foot_heights[kf,0]),
#             foot_R    = float(foot_heights[kf,1]),
#             weight    = lma_feats[idx]['weight'],
#             timeq     = lma_feats[idx]['time'],
#             spaceq    = lma_feats[idx]['space'],
#             flowq     = lma_feats[idx]['flow']
#         ))
#         segs.append(seg)
#     engine.run()
#     sync_reasoner([onto], infer_property_values=True)
#     return segs
    