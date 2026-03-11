from ctypes import WinDLL, create_string_buffer
import clr
import os
import sys
import time
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import random #debug
from khoi_custom_prior_interface import prior as prior
import sys

#Constant declaration
TEM_PID_MAX = 500
TEM_PID_MIN = -500

POS_MAX = 1000000000
POS_MIN = -1000000000

STEP_SIZE_MAX = 10000000
STEP_SIZE_MIN = -10000000

ACCEL_MAX = 10000000
ACCEL_MIN = -10000000

SPEED_MAX = 10000000
SPEED_MIN = -10000000

COEFF_SIZE_MAX = 1000000000
COEFF_SIZE_MIN = 0

BACKLASH_DIST_MIN = -1000000000
BACKLASH_DIST_MAX = 1000000000

# Window declaration
root = Tk() 
fig = Figure(figsize=(3,2), dpi = 85)

#Machine variable initialization (Change COM port here)
pr_off = False
try:
    pr = prior(sys.argv[2], os.getcwd() + r"\PriorSDK1.9.2\x64\PriorScientificSDK.dll")

    Prior_X_pos = pr.x
    Prior_Y_pos = pr.y
    Prior_XY_Speed = pr.velocity
    Prior_XY_Acceleration = pr.acceleration
    Prior_XY_Backlash_EN = IntVar(value=pr.backlash_en)
    Prior_XY_Backlash_Dist = pr.backlash_dist

    Prior_Z_More_Setting_displacement = 2
    Prior_Z_Speed = pr.z_velocity
    Prior_Z_Acceleration = pr.z_acceleration
    Prior_Z_Backlash_EN = IntVar(value=pr.backlash_en)
    Prior_Z_Backlash_Dist = pr.backlash_dist

except:
    pr_off = True
    print("Prior Controller is not connected")
    Prior_X_pos = 0
    Prior_Y_pos = 0
    Prior_XY_Speed = 0
    Prior_XY_Acceleration = 0
    Prior_XY_Backlash_EN = IntVar(value=0)
    Prior_XY_Backlash_Dist = 0

    Prior_Z_Speed = 0
    Prior_Z_Acceleration = 0
    Prior_Z_Backlash_EN = IntVar(value=0)
    Prior_Z_Backlash_Dist = 0

#Variable declaration ##################################

##Prior
Prior_XY_is_Con = False
Prior_Z_is_Con = False

Prior_XY_Step_size = 1
Prior_XY_coeff = 1
Prior_XY_More_Setting_displacement = 2


Prior_left_X_isHold = False
Prior_right_X_isHold = False
Prior_up_X_isHold = False
Prior_down_X_isHold = False

Prior_Z_More_Setting_displacement = 2

Prior_Z_Step_size = 1
Prior_Z_coeff = 1

Prior_Z_pos = 0 #debug variable

#Update functions ################################

##Prior
def Prior_update_X_pos_string(*args): #Check with Prior API, not global variable (i.e unfinished)
    global Prior_X_pos, pr, Prior_X_pos
    if pr_off:
        Prior_X_pos = 0
        Prior_X_pos_string.set(0)  
    else:
        pr.get_curr_pos()
        Prior_X_pos = pr.x
        Prior_X_pos_string.set(pr.x)
    # root.after(250, Prior_update_X_pos_string)

def Prior_update_Y_pos_string(*args): #Check with Prior API, not global variable (i.e unfinished)
    global Prior_Y_pos, pr, Prior_Y_pos
    if pr_off:
        Prior_Y_pos = 0
        Prior_Y_pos_string.set(0)
    else:
        pr.get_curr_pos()
        Prior_Y_pos = pr.y
        Prior_Y_pos_string.set(pr.y)
    # root.after(250, Prior_update_Y_pos_string)

def Prior_update_XY_Step_size():
    global Prior_XY_Step_size
    Prior_XY_Step_size_string.set(Prior_XY_Step_size_spinbox.get())
    if (Prior_XY_Step_size_spinbox.get() != ""):
        Prior_XY_Step_size = int(Prior_XY_Step_size_spinbox.get())
    print("Prior_XY_Step_size = ", Prior_XY_Step_size) #debug

def Prior_update_XY_Step_size_text(*args):
    global Prior_XY_Step_size, Prior_XY_Step_size_string
    print("Prior_XY_Step_size string = " + Prior_XY_Step_size_string.get()) #debug
    if (Prior_XY_Step_size_string.get() != ""):
        Prior_XY_Step_size = int(Prior_XY_Step_size_string.get())
    print("Prior Step (μm) size text = ", Prior_XY_Step_size) #debug

def Prior_update_XY_coeff():
    global Prior_XY_coeff, Prior_XY_coeff_spinbox
    Prior_XY_coeff_string.set(Prior_XY_coeff_spinbox.get())
    if (Prior_XY_coeff_spinbox.get() != ""):
        Prior_XY_coeff = int(Prior_XY_coeff_spinbox.get())
    print("Prior_XY_coeff = ", Prior_XY_coeff) #debug

def Prior_update_XY_coeff_text(*args):
    global Prior_XY_coeff, Prior_XY_coeff_string
    print("Prior_XY_coeff string = " + Prior_XY_coeff_string.get()) #debug
    if (Prior_XY_coeff_string.get() != ""):
        Prior_XY_coeff = int(Prior_XY_coeff_string.get())
    print("Prior_XY_coeff text = ", Prior_XY_coeff) #debug

def Prior_update_XY_Speed():
    global Prior_XY_Speed, Prior_XY_Speed_spinbox
    Prior_XY_Speed_string.set(Prior_XY_Speed_spinbox.get())
    if (Prior_XY_Speed_spinbox.get() != ""):
        Prior_XY_Speed = int(Prior_XY_Speed_spinbox.get())
        pr.set_velocity(Prior_XY_Speed)
    print("Prior_XY_Speed = ", Prior_XY_Speed) #debug

def Prior_update_XY_Speed_text(*args):
    global Prior_XY_Speed, Prior_XY_Speed_string
    print("Prior_XY_Speed string = " + Prior_XY_Speed_string.get()) #debug
    if (Prior_XY_Speed_string.get() != ""):
        Prior_XY_Speed = int(Prior_XY_Speed_string.get())
        pr.set_velocity(Prior_XY_Speed)
    print("Prior_XY_Speed text = ", Prior_XY_Speed) #debug

def Prior_update_XY_Acceleration():
    global Prior_XY_Acceleration, Prior_XY_Acceleration_spinbox
    Prior_XY_Acceleration_string.set(Prior_XY_Acceleration_spinbox.get())
    if (Prior_XY_Acceleration_spinbox.get() != ""):
        Prior_XY_Acceleration = int(Prior_XY_Acceleration_spinbox.get())
        pr.set_acceleration(Prior_XY_Acceleration)
    print("Prior_XY_Acceleration = ", Prior_XY_Acceleration) #debug

def Prior_update_XY_Acceleration_text(*args):
    global Prior_XY_Acceleration, Prior_XY_Acceleration_string
    print("Prior_XY_Acceleration string = " + Prior_XY_Acceleration_string.get()) #debug
    if (Prior_XY_Acceleration_string.get() != ""):
        Prior_XY_Acceleration = int(Prior_XY_Acceleration_string.get())
        pr.set_acceleration(Prior_XY_Acceleration)
    print("Prior_XY_Acceleration text = ", Prior_XY_Acceleration) #debug

def Prior_update_XY_Backlash_Enable():
    global pr, Prior_XY_Backlash_EN
    pr.set_backlash_en(Prior_XY_Backlash_EN.get())

def Prior_update_XY_Backlash_Dist():
    global Prior_XY_Backlash_Dist, Prior_XY_Backlash_Dist_spinbox
    Prior_XY_Backlash_Dist_string.set(Prior_XY_Backlash_Dist_spinbox.get())
    if (Prior_XY_Backlash_Dist_spinbox.get() != ""):
        Prior_XY_Backlash_Dist = int(Prior_XY_Backlash_Dist_spinbox.get())
        pr.set_backlash_dist(Prior_XY_Backlash_Dist)
    print("Prior_XY_Backlash_Dist = ", Prior_XY_Backlash_Dist) #debug

def Prior_update_XY_Backlash_Dist_text(*args):
    global Prior_XY_Backlash_Dist, Prior_XY_Backlash_Dist_string
    print("Prior_XY_Backlash_Dist string = " + Prior_XY_Backlash_Dist_string.get()) #debug
    if (Prior_XY_Backlash_Dist_string.get() != ""):
        Prior_XY_Backlash_Dist = int(Prior_XY_Backlash_Dist_string.get())
        pr.set_backlash_dist(Prior_XY_Backlash_Dist)
    print("Prior_XY_Backlash_Dist text = ", Prior_XY_Backlash_Dist) #debug

def Prior_up_Y_pos(*args):
    global Prior_Y_pos, Prior_XY_Step_size, pr, Prior_X_pos, Prior_XY_coeff
    Prior_Y_pos -= Prior_XY_Step_size * Prior_XY_coeff
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_x10_up_Y_pos(*args):
    global Prior_Y_pos, Prior_XY_Step_size, pr, Prior_X_pos, Prior_XY_coeff
    Prior_Y_pos -= Prior_XY_Step_size * Prior_XY_coeff * 10
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_down_Y_pos(*args):
    global Prior_Y_pos, Prior_XY_Step_size, pr, Prior_X_pos, Prior_XY_coeff
    Prior_Y_pos += Prior_XY_Step_size * Prior_XY_coeff
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_x10_down_Y_pos(*args):
    global Prior_Y_pos, Prior_XY_Step_size, pr, Prior_X_pos, Prior_XY_coeff
    Prior_Y_pos += Prior_XY_Step_size * Prior_XY_coeff * 10
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_right_X_pos(*args):
    global Prior_X_pos, Prior_XY_Step_size, pr, Prior_Y_pos, Prior_XY_coeff
    Prior_X_pos += Prior_XY_Step_size * Prior_XY_coeff
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_x10_right_X_pos(*args):
    global Prior_X_pos, Prior_XY_Step_size, pr, Prior_Y_pos, Prior_XY_coeff
    Prior_X_pos += Prior_XY_Step_size * Prior_XY_coeff * 10
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_left_X_pos(*args):
    global Prior_X_pos, Prior_XY_Step_size, pr, Prior_Y_pos, Prior_XY_coeff
    Prior_X_pos -= Prior_XY_Step_size * Prior_XY_coeff
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_x10_left_X_pos(*args):
    global Prior_X_pos, Prior_XY_Step_size, pr, Prior_Y_pos, Prior_XY_coeff
    Prior_X_pos -= Prior_XY_Step_size * Prior_XY_coeff * 10
    pr.go_to_pos(Prior_X_pos, Prior_Y_pos)
    Prior_update_X_pos_string()
    Prior_update_Y_pos_string()

def Prior_hold_right_X_pos(*args):
    global pr
    pr.start_forward_x_motor()

def Prior_release_X_pos(*args):
    global pr
    pr.stop_x_motor()
    Prior_update_X_pos_string()

def Prior_hold_left_X_pos(*args):
    global pr
    pr.start_backward_x_motor()

def Prior_hold_up_Y_pos(*args):
    global pr
    pr.start_forward_y_motor()

def Prior_release_Y_pos(*args):
    global pr
    pr.stop_y_motor()
    Prior_update_Y_pos_string()

def Prior_hold_down_Y_pos(*args):
    global pr
    pr.start_backward_y_motor()

def Prior_continuous_setup(*args):
    global Prior_Up_button, Prior_Down_button, Prior_Right_button, Prior_Left_button
    Prior_Up_button.unbind("<ButtonRelease-1>")
    Prior_Up_button.bind("<Button-1>", Prior_hold_up_Y_pos)
    Prior_Up_button.bind("<ButtonRelease-1>", Prior_release_Y_pos)

    Prior_Down_button.unbind("<ButtonRelease-1>")
    Prior_Down_button.bind("<Button-1>", Prior_hold_down_Y_pos)
    Prior_Down_button.bind("<ButtonRelease-1>", Prior_release_Y_pos)

    Prior_Right_button.unbind("<ButtonRelease-1>")
    Prior_Right_button.bind("<Button-1>", Prior_hold_right_X_pos)
    Prior_Right_button.bind("<ButtonRelease-1>", Prior_release_X_pos)

    Prior_Left_button.unbind("<ButtonRelease-1>")
    Prior_Left_button.bind("<Button-1>", Prior_hold_left_X_pos)
    Prior_Left_button.bind("<ButtonRelease-1>", Prior_release_X_pos)

    global Prior_Up_x10_button, Prior_Down_x10_button, Prior_Right_x10_button, Prior_Left_x10_button
    Prior_Up_x10_button.unbind("<ButtonRelease-1>")
    Prior_Up_x10_button.bind("<Button-1>", Prior_hold_up_Y_pos)
    Prior_Up_x10_button.bind("<ButtonRelease-1>", Prior_release_Y_pos)

    Prior_Down_x10_button.unbind("<ButtonRelease-1>")
    Prior_Down_x10_button.bind("<Button-1>", Prior_hold_down_Y_pos)
    Prior_Down_x10_button.bind("<ButtonRelease-1>", Prior_release_Y_pos)

    Prior_Right_x10_button.unbind("<ButtonRelease-1>")
    Prior_Right_x10_button.bind("<Button-1>", Prior_hold_right_X_pos)
    Prior_Right_x10_button.bind("<ButtonRelease-1>", Prior_release_X_pos)

    Prior_Left_x10_button.unbind("<ButtonRelease-1>")
    Prior_Left_x10_button.bind("<Button-1>", Prior_hold_left_X_pos)
    Prior_Left_x10_button.bind("<ButtonRelease-1>", Prior_release_X_pos)

def Prior_discreet_setup(*args):
    global Prior_Up_button, Prior_Down_button, Prior_Right_button, Prior_Left_button
    Prior_Up_button.unbind("<Button-1>")
    Prior_Up_button.unbind("<ButtonRelease-1>")
    Prior_Up_button.bind("<ButtonRelease-1>", Prior_up_Y_pos)

    Prior_Down_button.unbind("<Button-1>")
    Prior_Down_button.unbind("<ButtonRelease-1>")
    Prior_Down_button.bind("<ButtonRelease-1>", Prior_down_Y_pos)

    Prior_Right_button.unbind("<Button-1>")
    Prior_Right_button.unbind("<ButtonRelease-1>")
    Prior_Right_button.bind("<ButtonRelease-1>", Prior_right_X_pos)

    Prior_Left_button.unbind("<Button-1>")
    Prior_Left_button.unbind("<ButtonRelease-1>")
    Prior_Left_button.bind("<ButtonRelease-1>", Prior_left_X_pos)

    global Prior_Up_x10_button, Prior_Down_x10_button, Prior_Right_x10_button, Prior_Left_x10_button
    Prior_Up_x10_button.unbind("<Button-1>")
    Prior_Up_x10_button.unbind("<ButtonRelease-1>")
    Prior_Up_x10_button.bind("<ButtonRelease-1>", Prior_x10_up_Y_pos)

    Prior_Down_x10_button.unbind("<Button-1>")
    Prior_Down_x10_button.unbind("<ButtonRelease-1>")
    Prior_Down_x10_button.bind("<ButtonRelease-1>", Prior_x10_down_Y_pos)

    Prior_Right_x10_button.unbind("<Button-1>")
    Prior_Right_x10_button.unbind("<ButtonRelease-1>")
    Prior_Right_x10_button.bind("<ButtonRelease-1>", Prior_x10_right_X_pos)

    Prior_Left_x10_button.unbind("<Button-1>")
    Prior_Left_x10_button.unbind("<ButtonRelease-1>")
    Prior_Left_x10_button.bind("<ButtonRelease-1>", Prior_x10_left_X_pos)

def Prior_update_XY_modetoCon(*args):
    global Prior_XY_is_Con, Prior_Con_button, Prior_Dis_button
    if Prior_XY_is_Con == False:
        Prior_Con_button.configure(relief="sunken")
        Prior_Dis_button.configure(relief="raised")
        Prior_XY_is_Con = True
        Prior_continuous_setup()
    print("Prior_XY_is_Con = ", Prior_XY_is_Con)

def Prior_update_XY_modetoDis(*args):
    global Prior_XY_is_Con, Prior_Con_button, Prior_Dis_button
    if Prior_XY_is_Con == True:
        Prior_Con_button.configure(relief="raised")
        Prior_Dis_button.configure(relief="sunken")
        Prior_XY_is_Con = False
        Prior_discreet_setup()
    print("Prior_XY_is_Con = ", Prior_XY_is_Con)

# def Prior_update_XY_pos(*args):
#     global Prior_X_pos, Prior_Y_pos
#     if ((Prior_Im_X_pos_string.get() != "") & (Prior_Im_Y_pos_string.get() != "")):
#         Prior_X_pos = int(Prior_Im_X_pos_string.get())
#         Prior_Y_pos = int(Prior_Im_Y_pos_string.get())

def Prior_update_Z_modetoCon(*args):
    global Prior_Z_is_Con, Prior_Z_Con_button, Prior_Z_Dis_button
    if Prior_Z_is_Con == False:
        Prior_Z_Con_button.configure(relief="sunken")
        Prior_Z_Dis_button.configure(relief="raised")
        Prior_Z_is_Con = True
        Prior_Z_continuous_setup()
    print("Prior_Z_is_Con = ", Prior_Z_is_Con)

def Prior_update_Z_modetoDis(*args):
    global Prior_Z_is_Con, Prior_Z_Con_button, Prior_Z_Dis_button
    if Prior_Z_is_Con == True:
        Prior_Z_Con_button.configure(relief="raised")
        Prior_Z_Dis_button.configure(relief="sunken")
        Prior_Z_is_Con = False
        Prior_Z_discreet_setup()
    print("Prior_Z_is_Con = ", Prior_Z_is_Con)

def Prior_update_Z_pos_string(*args): #Check with Prior API, not global variable (i.e unfinished)
    global Prior_Z_pos, Prior_Z_pos_string, pr
    Prior_Z_pos = pr.get_curr_z_pos()
    Prior_Z_pos_string.set(Prior_Z_pos)

def Prior_update_Z_Step_size():
    global Prior_Z_Step_size
    Prior_Z_Step_size_string.set(Prior_Z_Step_size_spinbox.get())
    if (Prior_Z_Step_size_spinbox.get() != ""):
        Prior_Z_Step_size = int(Prior_Z_Step_size_spinbox.get())
    print("Prior_Z_Step_size = ", Prior_Z_Step_size) #debug

def Prior_update_Z_Step_size_text(*args):
    global Prior_Z_Step_size, Prior_Z_Step_size_string
    print("Prior_Z_Step_size string = " + Prior_Z_Step_size_string.get()) #debug
    if (Prior_Z_Step_size_string.get() != ""):
        Prior_Z_Step_size = int(Prior_Z_Step_size_string.get())
    print("Prior_Step size text = ", Prior_Z_Step_size) #debug

def Prior_update_Z_coeff():
    global Prior_Z_coeff, Prior_Z_coeff_spinbox
    Prior_Z_coeff_string.set(Prior_Z_coeff_spinbox.get())
    if (Prior_Z_coeff_spinbox.get() != ""):
        Prior_Z_coeff = int(Prior_Z_coeff_spinbox.get())
    print("Prior_Z_coeff = ", Prior_Z_coeff) #debug

def Prior_update_Z_coeff_text(*args):
    global Prior_Z_coeff, Prior_Z_coeff_string
    print("Prior_Z_coeff string = " + Prior_Z_coeff_string.get()) #debug
    if (Prior_Z_coeff_string.get() != ""):
        Prior_Z_coeff = int(Prior_Z_coeff_string.get())
    print("Prior_Z_coeff text = ", Prior_Z_coeff) #debug

def Prior_update_Z_Speed():
    global Prior_Z_Speed, Prior_Z_Speed_spinbox
    Prior_Z_Speed_string.set(Prior_Z_Speed_spinbox.get())
    if (Prior_Z_Speed_spinbox.get() != ""):
        Prior_Z_Speed = int(Prior_Z_Speed_spinbox.get())
        pr.set_z_velocity(Prior_Z_Speed)
    print("Prior_Z_Speed = ", Prior_Z_Speed) #debug

def Prior_update_Z_Speed_text(*args):
    global Prior_Z_Speed, Prior_Z_Speed_string
    print("Prior_Z_Speed string = " + Prior_Z_Speed_string.get()) #debug
    if (Prior_Z_Speed_string.get() != ""):
        Prior_Z_Speed = int(Prior_Z_Speed_string.get())
        pr.set_z_velocity(Prior_Z_Speed)
    print("Prior_Z_Speed text = ", Prior_Z_Speed) #debug

def Prior_update_Z_Acceleration():
    global Prior_Z_Acceleration, Prior_Z_Acceleration_spinbox
    Prior_Z_Acceleration_string.set(Prior_Z_Acceleration_spinbox.get())
    if (Prior_Z_Acceleration_spinbox.get() != ""):
        Prior_Z_Acceleration = int(Prior_Z_Acceleration_spinbox.get())
        pr.set_z_acceleration(Prior_Z_Acceleration)
    print("Prior_Z_Acceleration = ", Prior_Z_Acceleration) #debug

def Prior_update_Z_Acceleration_text(*args):
    global Prior_Z_Acceleration, Prior_Z_Acceleration_string
    print("Prior_Z_Acceleration string = " + Prior_Z_Acceleration_string.get()) #debug
    if (Prior_Z_Acceleration_string.get() != ""):
        Prior_Z_Acceleration = int(Prior_Z_Acceleration_string.get())
        pr.set_z_acceleration(Prior_Z_Acceleration)
    print("Prior_Z_Acceleration text = ", Prior_Z_Acceleration) #debug

def Prior_update_Z_Backlash_Enable():
    global pr, Prior_Z_Backlash_EN
    pr.set_z_backlash_en(Prior_Z_Backlash_EN.get())

def Prior_update_Z_Backlash_Dist():
    global Prior_Z_Backlash_Dist, Prior_Z_Backlash_Dist_spinbox
    Prior_Z_Backlash_Dist_string.set(Prior_Z_Backlash_Dist_spinbox.get())
    if (Prior_Z_Backlash_Dist_spinbox.get() != ""):
        Prior_Z_Backlash_Dist = int(Prior_Z_Backlash_Dist_spinbox.get())
        pr.set_z_backlash_dist(Prior_Z_Backlash_Dist)
    print("Prior_Z_Backlash_Dist = ", Prior_Z_Backlash_Dist) #debug

def Prior_update_Z_Backlash_Dist_text(*args):
    global Prior_Z_Backlash_Dist, Prior_Z_Backlash_Dist_string
    print("Prior_Z_Backlash_Dist string = " + Prior_Z_Backlash_Dist_string.get()) #debug
    if (Prior_Z_Backlash_Dist_string.get() != ""):
        Prior_Z_Backlash_Dist = int(Prior_Z_Backlash_Dist_string.get())
        pr.set_z_backlash_dist(Prior_Z_Backlash_Dist)
    print("Prior_Z_Backlash_Dist text = ", Prior_Z_Backlash_Dist) #debug

def Prior_up_Z_pos(*args):
    global Prior_Z_pos, Prior_Z_Step_size, Prior_Z_coeff
    Prior_Z_pos += Prior_Z_Step_size * Prior_Z_coeff
    pr.go_to_z_pos(Prior_Z_pos)
    Prior_update_Z_pos_string()


def Prior_x10_up_Z_pos(*args):
    global Prior_Z_pos, Prior_Z_Step_size, Prior_Z_coeff
    Prior_Z_pos += Prior_Z_Step_size * Prior_Z_coeff * 10
    pr.go_to_z_pos(Prior_Z_pos)
    Prior_update_Z_pos_string()

def Prior_down_Z_pos(*args):
    global Prior_Z_pos, Prior_Z_Step_size, Prior_Z_coeff
    Prior_Z_pos -= Prior_Z_Step_size * Prior_Z_coeff
    pr.go_to_z_pos(Prior_Z_pos)
    Prior_update_Z_pos_string()
    
def Prior_x10_down_Z_pos(*args):
    global Prior_Z_pos, Prior_Z_Step_size, Prior_Z_coeff
    Prior_Z_pos -= Prior_Z_Step_size * Prior_Z_coeff * 10
    pr.go_to_z_pos(Prior_Z_pos)
    Prior_update_Z_pos_string()

def Prior_hold_up_Z_pos(*args):
    global pr
    pr.start_forward_z_motor()

def Prior_release_Z_pos(*args):
    global pr
    pr.stop_z_motor()
    Prior_update_Z_pos_string()

def Prior_hold_down_Z_pos(*args):
    global pr
    pr.start_backward_z_motor()

def Prior_Z_continuous_setup(*args):
    global Prior_Z_Up_button, Prior_Z_Down_button
    Prior_Z_Up_button.unbind("<ButtonRelease-1>")
    Prior_Z_Up_button.bind("<Button-1>", Prior_hold_up_Z_pos)
    Prior_Z_Up_button.bind("<ButtonRelease-1>", Prior_release_Z_pos)

    Prior_Z_Down_button.unbind("<ButtonRelease-1>")
    Prior_Z_Down_button.bind("<Button-1>", Prior_hold_down_Z_pos)
    Prior_Z_Down_button.bind("<ButtonRelease-1>", Prior_release_Z_pos)

    global Prior_Z_Up_x10_button, Prior_Z_Down_x10_button
    Prior_Z_Up_x10_button.unbind("<ButtonRelease-1>")
    Prior_Z_Up_x10_button.bind("<Button-1>", Prior_hold_up_Z_pos)
    Prior_Z_Up_x10_button.bind("<ButtonRelease-1>", Prior_release_Z_pos)

    Prior_Z_Down_x10_button.unbind("<ButtonRelease-1>")
    Prior_Z_Down_x10_button.bind("<Button-1>", Prior_hold_down_Z_pos)
    Prior_Z_Down_x10_button.bind("<ButtonRelease-1>", Prior_release_Z_pos)

def Prior_Z_discreet_setup(*args):
    global Prior_Z_Up_button, Prior_Z_Down_button
    Prior_Z_Up_button.unbind("<Button-1>")
    Prior_Z_Up_button.unbind("<ButtonRelease-1>")
    Prior_Z_Up_button.bind("<ButtonRelease-1>", Prior_up_Z_pos)

    Prior_Z_Down_button.unbind("<Button-1>")
    Prior_Z_Down_button.unbind("<ButtonRelease-1>")
    Prior_Z_Down_button.bind("<ButtonRelease-1>", Prior_down_Z_pos)

    global Prior_Z_Up_x10_button, Prior_Z_Down_x10_button
    Prior_Z_Up_x10_button.unbind("<Button-1>")
    Prior_Z_Up_x10_button.unbind("<ButtonRelease-1>")
    Prior_Z_Up_x10_button.bind("<ButtonRelease-1>", Prior_x10_up_Z_pos)

    Prior_Z_Down_x10_button.unbind("<Button-1>")
    Prior_Z_Down_x10_button.unbind("<ButtonRelease-1>")
    Prior_Z_Down_x10_button.bind("<ButtonRelease-1>", Prior_x10_down_Z_pos)


# def Prior_update_Z_pos(*argss):
#     global Prior_Z_pos
#     if (Prior_Im_Z_pos_string.get() != ""):
#         Prior_Z_pos = int(Prior_Im_Z_pos_string.get())

def Prior_XY_hide_Setting(*args):
    global Prior_XY_More_Setting_displacement, Prior_XY_More_Setting_frame
    Prior_XY_More_Setting_displacement = 2
    Prior_XY_More_Setting_frame.grid_forget()

    Prior_Z_Label_seperator.grid(column=3, row=18-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ew")
    Prior_Z_control_label.grid(column=3, row=19-Prior_XY_More_Setting_displacement, columnspan=2, sticky="nsew")

    Prior_Z_pos_label.grid(column=3, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")
    Prior_Z_pos_textblock.grid(column=4, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")

    Prior_Z_button_frame.grid(column=3, row=21- Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2)

    Prior_Z_Setting_frame.grid(column=3, row=23-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ns")

    if (Prior_Z_More_Setting_displacement == 0):
        Prior_Z_More_Setting_frame.grid(column=3, row=24-Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2, sticky="ns")
    
def Prior_XY_show_Setting(*args):
    global Prior_XY_More_Setting_displacement, Prior_XY_More_Setting_frame
    Prior_XY_More_Setting_displacement = 0
    Prior_XY_More_Setting_frame.grid(column=3, row=16, columnspan=2, rowspan=2, sticky="ns")
    Prior_Z_Label_seperator.grid(column=3, row=18-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ew")
    Prior_Z_control_label.grid(column=3, row=19-Prior_XY_More_Setting_displacement, columnspan=2, sticky="nsew")

    Prior_Z_pos_label.grid(column=3, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")
    Prior_Z_pos_textblock.grid(column=4, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")

    Prior_Z_button_frame.grid(column=3, row=21- Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2)

    Prior_Z_Setting_frame.grid(column=3, row=23-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ns")

    if (Prior_Z_More_Setting_displacement == 0):
        Prior_Z_More_Setting_frame.grid(column=3, row=24-Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2, sticky="ns")
  
def Prior_XY_hide_show_Setting(*args):
    global Prior_XY_More_Setting_displacement
    if (Prior_XY_More_Setting_displacement == 0):
        Prior_XY_hide_Setting()
    else:
        Prior_XY_show_Setting()

def Prior_Z_hide_Setting(*args):
    global Prior_Z_More_Setting_displacement, Prior_Z_More_Setting_frame
    Prior_Z_More_Setting_displacement = 2
    Prior_Z_More_Setting_frame.grid_forget()

def Prior_Z_show_Setting(*args):
    global Prior_Z_More_Setting_displacement, Prior_Z_More_Setting_frame
    Prior_Z_More_Setting_displacement = 0
    Prior_Z_More_Setting_frame.grid(column=3, row=24-Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2, sticky="ns")

def Prior_Z_hide_show_Setting(*args):
    global Prior_Z_More_Setting_displacement
    if (Prior_Z_More_Setting_displacement == 0):
        Prior_Z_hide_Setting()
    else:
        Prior_Z_show_Setting()

def on_close():
    pr.disconnect()
    root.destroy()

# GUI Variable ################################
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

##Prior
Prior_X_pos_string = StringVar()
Prior_X_pos_string.set(Prior_X_pos)
Prior_Y_pos_string = StringVar()
Prior_Y_pos_string.set(Prior_Y_pos)

Prior_XY_Step_size_string = StringVar()
Prior_XY_Step_size_string.set(Prior_XY_Step_size)

Prior_XY_coeff_string = StringVar()
Prior_XY_coeff_string.set(Prior_XY_coeff)

Prior_XY_Speed_string = StringVar()
Prior_XY_Speed_string.set(Prior_XY_Speed)

Prior_XY_Acceleration_string = StringVar()
Prior_XY_Acceleration_string.set(Prior_XY_Acceleration)

Prior_XY_Backlash_Dist_string = StringVar()
Prior_XY_Backlash_Dist_string.set(Prior_XY_Backlash_Dist)

Prior_Z_pos_string = StringVar()
Prior_Z_pos_string.set(Prior_Z_pos)

Prior_Z_Step_size_string = StringVar()
Prior_Z_Step_size_string.set(Prior_Z_Step_size)

Prior_Z_coeff_string = StringVar()
Prior_Z_coeff_string.set(Prior_Z_coeff)

Prior_Z_Speed_string = StringVar()
Prior_Z_Speed_string.set(Prior_Z_Speed)

Prior_Z_Acceleration_string = StringVar()
Prior_Z_Acceleration_string.set(Prior_Z_Acceleration)

Prior_Z_Backlash_Dist_string = StringVar()
Prior_Z_Backlash_Dist_string.set(Prior_Z_Backlash_Dist)

# GUI Setting ###################################################
root.title("PriorThorLab")
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=8)
root.columnconfigure(2, weight=0)
root.columnconfigure(3, weight=10)
root.columnconfigure(4, weight=1)

root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)
root.rowconfigure(6, weight=1)
root.rowconfigure(7, weight=1)
root.rowconfigure(8, weight=1)
root.rowconfigure(9, weight=1)
root.rowconfigure(10, weight=1)
root.rowconfigure(11, weight=1)
root.rowconfigure(12, weight=1)
root.rowconfigure(13, weight=1)
root.rowconfigure(14, weight=1)
root.rowconfigure(15, weight=1)
root.rowconfigure(16, weight=1)
root.rowconfigure(17, weight=1)
root.rowconfigure(18, weight=1)
root.rowconfigure(19, weight=1)
root.rowconfigure(20, weight=1)
root.rowconfigure(21, weight=1)
root.rowconfigure(22, weight=1)
root.rowconfigure(23, weight=1)
root.rowconfigure(24, weight=1)
root.rowconfigure(25, weight=1)
root.rowconfigure(26, weight=1)
root.rowconfigure(27, weight=1)
root.rowconfigure(28, weight=1)
root.rowconfigure(29, weight=1)
root.rowconfigure(30, weight=1)
root.rowconfigure(31, weight=1)
root.rowconfigure(32, weight=1)

##TC200
TC_frame = Frame(root)

TC_frame.columnconfigure(0, weight=1)
TC_frame.columnconfigure(1, weight=1)

TC_frame.rowconfigure(0, weight=1)
TC_frame.rowconfigure(1, weight=1)
TC_frame.rowconfigure(2, weight=1)
TC_frame.rowconfigure(3, weight=1)
TC_frame.rowconfigure(4, weight=1)
TC_frame.rowconfigure(5, weight=1)
TC_frame.rowconfigure(6, weight=1)
TC_frame.rowconfigure(7, weight=1)

normal_font = "Helvetica 11"

##Prior
Prior_title = Label(root, text="ProScan Disconnected" if pr_off else "ProScan CONTROLLER", font="Helvetica 13")

Prior_XY_control_label = Label(root, text="XY AXIS CONTROL",font=normal_font)
Prior_X_pos_label = Label(root, text="X Position",font=normal_font)
Prior_X_pos_textblock = Label(root, borderwidth=1,textvariable=Prior_X_pos_string, relief="groove")

Prior_Y_pos_label = Label(root, text="Y Position",font=normal_font)
Prior_Y_pos_textblock = Label(root, borderwidth=1,textvariable=Prior_Y_pos_string, relief="groove")

Prior_XY_Setting_frame = Frame(root)

Prior_Setting_button = Button(Prior_XY_Setting_frame, text="Speed Setting", command=Prior_XY_hide_show_Setting)
Prior_XY_Step_size_label = Label(Prior_XY_Setting_frame, text="Step (μm)",font=normal_font)
Prior_XY_Step_size_spinbox = Spinbox(Prior_XY_Setting_frame, textvariable=Prior_XY_Step_size_string, from_=STEP_SIZE_MIN, to=STEP_SIZE_MAX, command=Prior_update_XY_Step_size, width=10)
Prior_XY_Step_size_string.trace_add("write", Prior_update_XY_Step_size_text)

Prior_XY_More_Setting_frame = Frame(root)

Prior_XY_coeff_label = Label(Prior_XY_More_Setting_frame, text="Multiplier")
Prior_XY_coeff_spinbox = Spinbox(Prior_XY_More_Setting_frame, textvariable=Prior_XY_coeff_string, from_=COEFF_SIZE_MIN, to=COEFF_SIZE_MAX, width=10, command=Prior_update_XY_coeff)
Prior_XY_coeff_string.trace_add("write", Prior_update_XY_coeff_text)

Prior_XY_Speed_label = Label(Prior_XY_More_Setting_frame, text="Speed (μm/s)")
Prior_XY_Speed_spinbox = Spinbox(Prior_XY_More_Setting_frame, textvariable=Prior_XY_Speed_string, from_=SPEED_MIN, to=SPEED_MAX, command=Prior_update_XY_Speed)
Prior_XY_Speed_string.trace_add("write", Prior_update_XY_Speed_text)

Prior_XY_Acceleration_label = Label(Prior_XY_More_Setting_frame, text="Accel (μm/s²)")
Prior_XY_Acceleration_spinbox = Spinbox(Prior_XY_More_Setting_frame, textvariable=Prior_XY_Acceleration_string, from_=ACCEL_MIN, to=ACCEL_MAX, command=Prior_update_XY_Acceleration)
Prior_XY_Acceleration_string.trace_add("write", Prior_update_XY_Acceleration_text)

Prior_XY_Backlash_EN_checkbox = Checkbutton(Prior_XY_More_Setting_frame, variable=Prior_XY_Backlash_EN, text="Backlash Enable", onvalue=1, offvalue=0, command=Prior_update_XY_Backlash_Enable)

Prior_XY_Backlash_Dist_label = Label(Prior_XY_More_Setting_frame, text="Backlash Dist (μm)")
Prior_XY_Backlash_Dist_spinbox = Spinbox(Prior_XY_More_Setting_frame, textvariable=Prior_XY_Backlash_Dist_string, from_=BACKLASH_DIST_MIN, to=BACKLASH_DIST_MAX, command=Prior_update_XY_Backlash_Dist)
Prior_XY_Backlash_Dist_string.trace_add("write", Prior_update_XY_Backlash_Dist_text)

Prior_button_frame = Frame(root)

Prior_Left_button = Button(Prior_button_frame, text="◄", font=5, width=3, height=1)
Prior_Right_button = Button(Prior_button_frame, text="►", font=5, width=3, height=1)
Prior_Up_button = Button(Prior_button_frame, text="▲", font=5, width=3, height=1)
Prior_Down_button = Button(Prior_button_frame, text="▼", font=5, width=3, height=1)

Prior_Left_x10_button = Button(Prior_button_frame, text="⏪")
Prior_Right_x10_button = Button(Prior_button_frame, text="⏩")
Prior_Up_x10_button = Button(Prior_button_frame, text="⏫")
Prior_Down_x10_button = Button(Prior_button_frame, text="⏬")

Prior_discreet_setup()

Prior_Con_button = Button(Prior_button_frame, text="Con", width=3, command=Prior_update_XY_modetoCon)
Prior_Dis_button = Button(Prior_button_frame, text="Jog", width=3, relief="sunken",command=Prior_update_XY_modetoDis)

Prior_Z_Label_seperator = ttk.Separator(root, orient="horizontal")
Prior_Z_control_label = Label(root, text="Z AXIS CONTROL",font=normal_font)

Prior_Z_pos_label = Label(root, text="Z Position",font=normal_font)
Prior_Z_pos_textblock = Label(root, borderwidth=1, textvariable=Prior_Z_pos_string, relief="groove")

Prior_Z_button_frame = Frame(root)

Prior_Z_Up_button = Button(Prior_Z_button_frame, text="▲", width=4, height=2)
Prior_Z_Down_button = Button(Prior_Z_button_frame, text="▼", width=4, height=2)

Prior_Z_Up_x10_button = Button(Prior_Z_button_frame, text="⏫",width=4, height=2)
Prior_Z_Down_x10_button = Button(Prior_Z_button_frame, text="⏬",width=4, height=2)

Prior_Z_discreet_setup()

Prior_Z_filler = Label(Prior_Z_button_frame, text="")

Prior_Z_Con_button = Button(Prior_Z_button_frame, text="Con", width=3, command=Prior_update_Z_modetoCon)
Prior_Z_Dis_button = Button(Prior_Z_button_frame, text="Jog", width=3, relief="sunken", command=Prior_update_Z_modetoDis)

Prior_Z_Setting_frame = Frame(root)

Prior_Z_Step_size_label = Label(Prior_Z_Setting_frame, text="Step (μm)",font=normal_font)
Prior_Z_Step_size_spinbox = Spinbox(Prior_Z_Setting_frame, textvariable=Prior_Z_Step_size_string, from_=STEP_SIZE_MIN, to=STEP_SIZE_MAX, command=Prior_update_Z_Step_size, width=10)
Prior_Z_Step_size_string.trace_add("write", Prior_update_Z_Step_size_text)
Prior_Z_Setting_button = Button(Prior_Z_Setting_frame, text="Speed Setting", command=Prior_Z_hide_show_Setting)

Prior_Z_More_Setting_frame = Frame(root)

Prior_Z_coeff_label = Label(Prior_Z_More_Setting_frame, text="Multiplier")
Prior_Z_coeff_spinbox = Spinbox(Prior_Z_More_Setting_frame, textvariable=Prior_Z_coeff_string, from_=COEFF_SIZE_MIN, to=COEFF_SIZE_MAX, width=10, command=Prior_update_Z_coeff)
Prior_Z_coeff_string.trace_add("write", Prior_update_Z_coeff_text)

Prior_Z_Speed_label = Label(Prior_Z_More_Setting_frame, text="Speed (μm/s)")
Prior_Z_Speed_spinbox = Spinbox(Prior_Z_More_Setting_frame, textvariable=Prior_Z_Speed_string, from_=SPEED_MIN, to=SPEED_MAX, command=Prior_update_Z_Speed)
Prior_Z_Speed_string.trace_add("write", Prior_update_Z_Speed_text)

Prior_Z_Acceleration_label = Label(Prior_Z_More_Setting_frame, text="Accel (μm/s²)")
Prior_Z_Acceleration_spinbox = Spinbox(Prior_Z_More_Setting_frame, textvariable=Prior_Z_Acceleration_string, from_=ACCEL_MIN, to=ACCEL_MAX, command=Prior_update_Z_Acceleration)
Prior_Z_Acceleration_string.trace_add("write", Prior_update_Z_Acceleration_text)

Prior_Z_Backlash_EN_checkbox = Checkbutton(Prior_Z_More_Setting_frame, variable=Prior_Z_Backlash_EN, text="Backlash Enable", onvalue=1, offvalue=0, command=Prior_update_Z_Backlash_Enable)

Prior_Z_Backlash_Dist_label = Label(Prior_Z_More_Setting_frame, text="Backlash Dist (μm)")
Prior_Z_Backlash_Dist_spinbox = Spinbox(Prior_Z_More_Setting_frame, textvariable=Prior_Z_Backlash_Dist_string, from_=BACKLASH_DIST_MIN, to=BACKLASH_DIST_MAX, command=Prior_update_Z_Backlash_Dist)
Prior_Z_Backlash_Dist_string.trace_add("write", Prior_update_Z_Backlash_Dist_text)

#GUI Placement ######################################################
root.grid_propagate(True)

##Prior
Prior_title.grid(column=3, row=9, columnspan=2, sticky="nsew")

Prior_XY_control_label.grid(column=3, row=10, columnspan=2, sticky="nsew")

Prior_X_pos_label.grid(column=3, row=11, sticky="nsew")
Prior_X_pos_textblock.grid(column=4, row=11, sticky="nsew")

Prior_Y_pos_label.grid(column=3, row=12, sticky="nsew")
Prior_Y_pos_textblock.grid(column=4, row=12, sticky="nsew")

Prior_button_frame.grid(column=3, row=13, rowspan=2, columnspan=2)

Prior_Up_button.grid(column=2, row=1, sticky="nsew")
Prior_Up_x10_button.grid(column=2, row=0, sticky="nsew")

Prior_Down_button.grid(column=2, row=3, sticky="nsew")
Prior_Down_x10_button.grid(column=2, row=4, sticky="nsew")

Prior_Right_button.grid(column=3, row=2, sticky="nsew")
Prior_Right_x10_button.grid(column=4, row=2, sticky="nsew")

Prior_Left_button.grid(column=1, row=2, sticky="nsew")
Prior_Left_x10_button.grid(column=0, row=2, sticky="nsew")

Prior_Con_button.grid(column=3, row=4, sticky="e")
Prior_Dis_button.grid(column=4, row=4, sticky="w")

Prior_XY_Setting_frame.grid(column=3, row=15, columnspan=2, sticky="ns")

Prior_Setting_button.grid(column=2, row=0, columnspan=2, sticky="nsew")
Prior_XY_Step_size_label.grid(column=0, row=0, sticky="nsew")
Prior_XY_Step_size_spinbox.grid(column=1, row=0, sticky="nsew")

Prior_XY_coeff_label.grid(column=0, row=0 ,sticky="nsew")
Prior_XY_coeff_spinbox.grid(column=1, row=0, sticky="nsew")

Prior_XY_Speed_label.grid(column=0, row=1, sticky="nsew")
Prior_XY_Speed_spinbox.grid(column=1, row=1, sticky="nsew")

Prior_XY_Acceleration_label.grid(column=0, row=2, sticky="nsew")
Prior_XY_Acceleration_spinbox.grid(column=1, row=2, sticky="nsew")

Prior_XY_Backlash_EN_checkbox.grid(column=0, columnspan=2, row=3, sticky="nsew")

Prior_XY_Backlash_Dist_label.grid(column=0, row=4, sticky="nsew")
Prior_XY_Backlash_Dist_spinbox.grid(column=1, row=4, sticky="nsew")

Prior_Z_Label_seperator.grid(column=3, row=18-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ew")
Prior_Z_control_label.grid(column=3, row=19-Prior_XY_More_Setting_displacement, columnspan=2, sticky="nsew")

Prior_Z_pos_label.grid(column=3, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")
Prior_Z_pos_textblock.grid(column=4, row=20-Prior_XY_More_Setting_displacement, sticky="nsew")

Prior_Z_button_frame.grid(column=3, row=21- Prior_XY_More_Setting_displacement, columnspan=2, rowspan=2)

Prior_Z_Up_button.grid(column=0, row=0)
Prior_Z_Down_button.grid(column=0, row=1)

Prior_Z_Up_x10_button.grid(column=1, row=0)
Prior_Z_Down_x10_button.grid(column=1, row=1)

Prior_Z_filler.grid(column=2, row=1, padx=7)
Prior_Z_Con_button.grid(column=3, row=1, sticky="s")
Prior_Z_Dis_button.grid(column=4, row=1, sticky="s")

Prior_Z_Setting_frame.grid(column=3, row=23-Prior_XY_More_Setting_displacement, columnspan=2, sticky="ns")

Prior_Z_Step_size_label.grid(column=0, row=0, sticky="nsew")
Prior_Z_Step_size_spinbox.grid(column=1, row=0, sticky="nsew")
Prior_Z_Setting_button.grid(column=2, columnspan=2, row=0, sticky="nsew")

# Prior_Z_More_Setting_frame.grid(column=3, row=23, columnspan=2, rowspan=2, sticky="ns")
Prior_Z_coeff_label.grid(column=0, row=0, sticky="nsew")
Prior_Z_coeff_spinbox.grid(column=1, row=0, sticky="nsew")

Prior_Z_Speed_label.grid(column=0, row=1, sticky="nsew")
Prior_Z_Speed_spinbox.grid(column=1, row=1, sticky="nsew")

Prior_Z_Acceleration_label.grid(column=0, row=2, sticky="nsew")
Prior_Z_Acceleration_spinbox.grid(column=1, row=2, sticky="nsew")

Prior_Z_Backlash_EN_checkbox.grid(column=0, row=3, columnspan=2, sticky="nsew")

Prior_Z_Backlash_Dist_label.grid(column=0, row=4, sticky="nsew")
Prior_Z_Backlash_Dist_spinbox.grid(column=1, row=4, sticky="nsew")


#Variable update call
Prior_update_X_pos_string()
Prior_update_Y_pos_string()
Prior_update_Z_pos_string()

#Calling Tk mainloop
root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()