import torch
import numpy as np

"""
One cycle of the leg rotation before the phase shift is in general: foot steps on the ground, 
                                                                    upper and lower legs rotate backward in sync, 
                                                                    foot leaves the ground, 
                                                                    upper and lower leg rotate forward in sync, 
                                                                    lower leg rotates backward,
                                                                    foot steps on the ground.
                                                                    during which, there might be rotation around x-axis
                                                                    so that the legs fold sidewides. 
Thus, the cycle of the leg before the phase shift always start as the foot steps on the ground.

where the first dimension indicates (k=12): 0-fl_hipx, 1-fl_hipy 2-fl_knee, 
                                            3-fr_hipx, 4-fr_hipy, 5-fr_knee, 
                                            6-hl_hipx, 7-hl_hipy, 8-hl_knee, 
                                            9-hr_hipx, 10=hr_hipy, 11-hr_knee

The coordinate system is defined as follows: 
  the x axis is always pointing along the body of the spot and toward the front,
  the z axis is alwyas pointing upward vertically,
  the y axis is determined by the right-hand rule, which is pointing to the left of the spot.

The hip joint x rotates around the x axis and its sign follows the right hand rule,
which means that the if the left legs rotate outward (inward) away (toward) from (to) the body center,
the angle is positive (negative).

The hip joint y rotates around the y axis and its sign follows the right hand rule,
which means that if the upper legs rotates backward (forward),
the angle is poistive (negative).

The knee joint has the same rule as the hip joint y.

The angle 0 for the hip joint x is defiend as the plane expanded by the upper leg and lower leg
is perpendicular to the ground.

The angle 0 for the hip joint y is defined as the upper leg pointing downwards to the ground.

The angle 0 for the knee joint is defined as the lower leg perfect align with the upper leg.
"""

hipx_ind_list = [0, 3, 6, 9]
hipy_ind_list = [1, 4, 7, 10]
knee_ind_list = [2, 5, 8, 11]
low_z = -0.3
high_z = -0.25

def get_angle_knee(angle_hipy, bone_length_uleg, bone_length_lleg, body_z, ground_z):
    distance_to_ground_knee = (body_z - ground_z) - torch.cos(angle_hipy) * bone_length_uleg
    angle_knee = -torch.acos(distance_to_ground_knee / bone_length_lleg) # always bend the knee backward.
    angle_knee = angle_knee - angle_hipy # get the relative angle with respect to the joint

    return angle_knee

def get_body_movement_z(low_z, high_z, cycle_length):
    z_incr = torch.linspace(low_z, high_z, int(cycle_length/2))
    z_decr = torch.linspace(high_z, low_z, cycle_length-int(cycle_length/2))
    z_cycle = torch.cat((z_incr, z_decr))

    return z_cycle

def walk_gait_spot(n, bone_length_uleg, bone_length_lleg, ground_z):
    body_z_one_cycle = -0.25*torch.ones(n) # This is specifically for the simulations on VisPy.
    joint_angles_one_cycle = torch.zeros(12, n)

    cycle_point10 = int(np.round(n * 6 / 10))
    cycle_point11 = int(np.round(n * 7 / 10))
    joint_angles_one_cycle[hipy_ind_list, :cycle_point10] = torch.linspace(torch.pi*2/12, torch.pi*5/12, cycle_point10)
    joint_angles_one_cycle[hipy_ind_list, cycle_point10:cycle_point11] = torch.linspace(torch.pi*5/12, torch.pi*5.5/12, cycle_point11-cycle_point10)
    joint_angles_one_cycle[hipy_ind_list, cycle_point11:] = torch.linspace(torch.pi*5.5/12, torch.pi*2/12, n-cycle_point11)

    cycle_point20 = int(np.round(n * 6 / 10))
    cycle_point21 = int(np.round(n * 8 / 10))
    joint_angles_one_cycle[knee_ind_list, :cycle_point20] = get_angle_knee(joint_angles_one_cycle[hipy_ind_list[0], :cycle_point20], bone_length_uleg, bone_length_lleg, body_z_one_cycle[:cycle_point20], ground_z)
    joint_angles_one_cycle[knee_ind_list, cycle_point20:cycle_point21] = torch.linspace(joint_angles_one_cycle[knee_ind_list[0], cycle_point20-1], -torch.pi*9/12, cycle_point21-cycle_point20)
    joint_angles_one_cycle[knee_ind_list, cycle_point21:] = torch.linspace(-torch.pi*9/12, joint_angles_one_cycle[knee_ind_list[0], 0], n-cycle_point21)

    ####### Shift the phases to get gait patterns.
    phase_shift = (torch.tensor([0.0, 0.5, 0.75, 0.25]) * n).type(torch.int)
    for ll in range(12):
        joint_angles_one_cycle[ll] = torch.roll(joint_angles_one_cycle[ll], phase_shift[ll//3].item(), dims=-1)

    return joint_angles_one_cycle, body_z_one_cycle

def trot_gait_spot(n, bone_length_uleg, bone_length_lleg, ground_z):
    body_z_one_cycle = -0.3*torch.ones(n) # This is specifically for the simulations on VisPy.
    joint_angles_one_cycle = torch.zeros(12, n)

    cycle_point10 = int(np.round(n * 5 / 10))
    cycle_point11 = int(np.round(n * 6 / 10))
    joint_angles_one_cycle[hipy_ind_list, :cycle_point10] = torch.linspace(torch.pi*2/12, torch.pi*5/12, cycle_point10)
    joint_angles_one_cycle[hipy_ind_list, cycle_point10:cycle_point11] = torch.linspace(torch.pi*5/12, torch.pi*5.5/12, cycle_point11-cycle_point10)
    joint_angles_one_cycle[hipy_ind_list, cycle_point11:] = torch.linspace(torch.pi*5.5/12, torch.pi*2/12, n-cycle_point11)

    cycle_point20 = int(np.round(n * 5 / 10))
    cycle_point21 = int(np.round(n * 7 / 10))
    body_z_one_cycle[:cycle_point20] = get_body_movement_z(low_z, high_z, cycle_point20)
    body_z_one_cycle[cycle_point20:] = get_body_movement_z(low_z, high_z, n-cycle_point20)
    joint_angles_one_cycle[knee_ind_list, :cycle_point20] = get_angle_knee(joint_angles_one_cycle[hipy_ind_list[0], :cycle_point20], bone_length_uleg, bone_length_lleg, body_z_one_cycle[:cycle_point20], ground_z)
    joint_angles_one_cycle[knee_ind_list, cycle_point20:cycle_point21] = torch.linspace(joint_angles_one_cycle[knee_ind_list[0], cycle_point20-1], -torch.pi*9/12, cycle_point21-cycle_point20)
    joint_angles_one_cycle[knee_ind_list, cycle_point21:] = torch.linspace(-torch.pi*9/12, joint_angles_one_cycle[knee_ind_list[0], 0], n-cycle_point21)

    ####### Shift the phases to get gait patterns.
    phase_shift = (torch.tensor([0.0, 0.5, 0.5, 0.0]) * n).type(torch.int)
    for ll in range(12):
        joint_angles_one_cycle[ll] = torch.roll(joint_angles_one_cycle[ll], phase_shift[ll//3].item(), dims=-1)

    return joint_angles_one_cycle, body_z_one_cycle

def gallop_gait_spot(n, bone_length_uleg, bone_length_lleg, ground_z):
    body_z_one_cycle = -0.3*torch.ones(n) # This body height in the z axis is fixed in the walk gait.
    joint_angles_one_cycle = torch.zeros(12, n)

    return joint_angles_one_cycle, body_z_one_cycle
        
        




















