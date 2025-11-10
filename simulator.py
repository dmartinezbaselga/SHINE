import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.animation as animation
import os
import struct

class Position:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

def read_scenario(batch_id, batch_i, dir):
    path = dir + "/scenario_" + str(batch_id) + "_" + str(batch_i) + ".bin"
    if os.path.exists(path):
        with open(path, "rb") as f:
            # Obstacles
            data = f.read(4)
            vec_length = struct.unpack("i", data)[0]
            obstacles_previous = []
            # Previous positions
            for _ in range(vec_length):
                pos_x = []
                pos_y = []
                obstacle_pos = []
                data = f.read(4)
                len_his = struct.unpack("i", data)[0]
                for _ in range(len_his):
                    data = f.read(8)
                    pos_x.append(struct.unpack("d", data)[0])
                for _ in range(len_his):
                    data = f.read(8)
                    pos_y.append(struct.unpack("d", data)[0])
                for i in range(len_his):
                    obstacle_pos.append(Position(pos_x[i], pos_y[i]))
                obstacles_previous.append(obstacle_pos)
            # Future positions
            obstacles_vec = []
            for _ in range(vec_length):
                pos_x = []
                pos_y = []
                obstacle_pos = []
                data = f.read(4)
                len_his = struct.unpack("i", data)[0]
                for _ in range(len_his):
                    data = f.read(8)
                    pos_x.append(struct.unpack("d", data)[0])
                for _ in range(len_his):
                    data = f.read(8)
                    pos_y.append(struct.unpack("d", data)[0])
                for i in range(len_his):
                    obstacle_pos.append(Position(pos_x[i], pos_y[i]))
                obstacles_vec.append(obstacle_pos)
            # x
            data = f.read(8)
            x = struct.unpack("d", data)[0]
            # y
            data = f.read(8)
            y = struct.unpack("d", data)[0]
            # orientation
            data = f.read(8)
            theta = struct.unpack("d", data)[0]
            # v
            data = f.read(8)
            v = struct.unpack("d", data)[0]
            # goals
            goals = []
            pos_x = []
            pos_y = []
            data = f.read(4)
            len_goals = struct.unpack("i", data)[0]
            for _ in range(len_goals):
                data = f.read(8)
                pos_x.append(struct.unpack("d", data)[0])
            for _ in range(len_goals):
                data = f.read(8)
                pos_y.append(struct.unpack("d", data)[0])
            for i in range(len_goals):
                goals.append(Position(pos_x[i], pos_y[i]))
            # human truth
            human_truth = []
            pos_x = []
            pos_y = []
            data = f.read(4)
            len_his = struct.unpack("i", data)[0]
            for _ in range(len_his):
                data = f.read(8)
                pos_x.append(struct.unpack("d", data)[0])
            for _ in range(len_his):
                data = f.read(8)
                pos_y.append(struct.unpack("d", data)[0])
            for i in range(len_his):
                human_truth.append(Position(pos_x[i], pos_y[i]))
            # Baseline traj
            baseline_traj = []
            pos_x = []
            pos_y = []
            data = f.read(4)
            len_his = struct.unpack("i", data)[0]
            for _ in range(len_his):
                data = f.read(8)
                pos_x.append(struct.unpack("d", data)[0])
            for _ in range(len_his):
                data = f.read(8)
                pos_y.append(struct.unpack("d", data)[0])
            for i in range(len_his):
                baseline_traj.append(Position(pos_x[i], pos_y[i]))
            # Ours traj
            ours_traj = []
            pos_x = []
            pos_y = []
            for _ in range(len_his):
                data = f.read(8)
                pos_x.append(struct.unpack("d", data)[0])
            for _ in range(len_his):
                data = f.read(8)
                pos_y.append(struct.unpack("d", data)[0])
            for i in range(len_his):
                ours_traj.append(Position(pos_x[i], pos_y[i]))
            # human previous
            human_previous = []
            pos_x = []
            pos_y = []
            data = f.read(4)
            len_his = struct.unpack("i", data)[0]
            for _ in range(len_his):
                data = f.read(8)
                pos_x.append(struct.unpack("d", data)[0])
            for _ in range(len_his):
                data = f.read(8)
                pos_y.append(struct.unpack("d", data)[0])
            for i in range(len_his):
                human_previous.append(Position(pos_x[i], pos_y[i]))
        return obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj
    else:
        return None
    
def init_plot(ax, id_scenario, person_radius, obstacles_previous, goals, human_previous):
    ax.cla()
    global obstacle
    global goal_artist
    obstacles_artist = []
    obstacles_arrow_artist = []
    for i_obs in range(len(obstacles_previous)):
        obstacle = plt.Circle((obstacles_previous[i_obs][0].x, obstacles_previous[i_obs][0].y), person_radius, color='black', fill=True)
        ax.add_artist(obstacle)
        obstacles_artist.append(obstacle)
        theta = 0.0
        obstacles_arrow_artist.append(plt.arrow(obstacles_previous[i_obs][0].x, obstacles_previous[i_obs][0].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red"))
    
    for i_goal in range(len(goals)):
        goal_artist = mlines.Line2D([goals[i_goal].x], [goals[i_goal].y],
                            color='red', marker='*', linestyle='None',
                            markersize=15, label='Goal')
        ax.add_artist(goal_artist)

    ax.title.set_text(r'Scenario {0}. Time: {1:.2f} s'.format(id_scenario, 0))
    robot_artist = plt.Circle((human_previous[0].x, human_previous[0].y), person_radius, fill=True, color="blue")
    ax.add_artist(robot_artist)
    theta = 0.0
    robot_arrow_artist = plt.arrow(human_previous[0].x, human_previous[0].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")
    # This is for plotting the trajectory
    ax.add_artist(plt.Circle((human_previous[0].x, human_previous[0].y), person_radius/8, fill=True, color="blue"))

    ax.add_artist(goal_artist)
    ax.legend([robot_artist, obstacle, goal_artist], ['Robot', 'Obstacle', 'Goal'], loc='lower right', markerscale=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    ax.axis('equal')
    # plt.tight_layout()
    time = 0
    return obstacles_artist, obstacles_arrow_artist, robot_artist, robot_arrow_artist, time


scenario_i = -1
batch_id = 0

person_radius = 0.25
timestep = 0.4

plt.close('all')
fig, ax = plt.subplots(1,1, figsize=(12,12))
first_iteration = True
# ax = ax.reshape((-1))
def update(t):
    global robot_arrow_artist, robot_ours, robot_baseline, obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj, obstacles_artist, obstacles_arrow_artist, robot_artist, robot_arrow_artist, first_iteration, scenario_i, batch_id
    if first_iteration:
        i_video = 0
        first_iteration = False
    else:
        i_video = t%(len(human_previous)+len(ours_traj))
    time = i_video*timestep
    if i_video > 0 and i_video < len(human_previous):
        time = i_video*timestep
        for i_obs in range(len(obstacles_previous)):
            obstacles_artist[i_obs].center = (obstacles_previous[i_obs][i_video].x, obstacles_previous[i_obs][i_video].y)
            obstacles_arrow_artist[i_obs].remove()
            theta = np.arctan2(obstacles_previous[i_obs][i_video].y - obstacles_previous[i_obs][i_video-1].y,
                               obstacles_previous[i_obs][i_video].x - obstacles_previous[i_obs][i_video-1].x)
            obstacles_arrow_artist[i_obs] = plt.arrow(obstacles_previous[i_obs][i_video].x, obstacles_previous[i_obs][i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")

        robot_artist.center = (human_previous[i_video].x, human_previous[i_video].y)
        ax.add_artist(plt.Circle((human_previous[i_video].x, human_previous[i_video].y), person_radius/8, fill=True, color="blue"))
        theta = np.arctan2(human_previous[i_video].y - human_previous[i_video-1].y,
                           human_previous[i_video].x - human_previous[i_video-1].x)
        # robot_arrow_artist.center
        robot_arrow_artist.remove()
        robot_arrow_artist = plt.arrow(human_previous[i_video].x, human_previous[i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")
        # This is for plotting the trajectory
        # ax.add_artist(plt.Circle((robot.x, robot.y), robot.radius/8, fill=True, color="blue"))
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-4, 4)
        # ax.axis('equal')
        ax.title.set_text(r'Scenario {0}. Time: {1:.2f} s'.format(scenario_i, time))
    elif i_video > 0:
        time = (i_video-1)*timestep
        i_video = i_video-len(human_previous)
        if i_video != 0:
            for i_obs in range(len(obstacles_vec)):
                obstacles_artist[i_obs].center = (obstacles_vec[i_obs][i_video].x, obstacles_vec[i_obs][i_video].y)
                obstacles_arrow_artist[i_obs].remove()
                theta = np.arctan2(obstacles_vec[i_obs][i_video].y - obstacles_vec[i_obs][i_video-1].y,
                                obstacles_vec[i_obs][i_video].x - obstacles_vec[i_obs][i_video-1].x)
                obstacles_arrow_artist[i_obs] = plt.arrow(obstacles_vec[i_obs][i_video].x, obstacles_vec[i_obs][i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")

            robot_ours.center = (ours_traj[i_video].x, ours_traj[i_video].y)
            ax.add_artist(plt.Circle((ours_traj[i_video].x, ours_traj[i_video].y), person_radius/8, fill=True, color="green"))
            theta = np.arctan2(ours_traj[i_video].y - ours_traj[i_video-1].y,
                                ours_traj[i_video].x - ours_traj[i_video-1].x)
            # robot_arrow_artist.center
            robot_arrow_artist.remove()
            robot_arrow_artist = plt.arrow(ours_traj[i_video].x, ours_traj[i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")
            # Baseline
            robot_baseline.center = (baseline_traj[i_video].x, baseline_traj[i_video].y)
            ax.add_artist(plt.Circle((baseline_traj[i_video].x, baseline_traj[i_video].y), person_radius/8, fill=True, color="red"))
            theta = np.arctan2(baseline_traj[i_video].y - baseline_traj[i_video-1].y,
                                baseline_traj[i_video].x - baseline_traj[i_video-1].x)
            # robot_arrow_artist.center
            robot_arrow_artist.remove()
            robot_arrow_artist = plt.arrow(ours_traj[i_video].x, ours_traj[i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")
 
            # Ground truth
            robot_artist.center = (human_truth[i_video].x, human_truth[i_video].y)
            ax.add_artist(plt.Circle((human_truth[i_video].x, human_truth[i_video].y), person_radius/8, fill=True, color="blue"))
            theta = np.arctan2(human_truth[i_video].y - human_truth[i_video-1].y,
                            human_truth[i_video].x - human_truth[i_video-1].x)
            # robot_arrow_artist.center
            robot_arrow_artist.remove()
            robot_arrow_artist = plt.arrow(human_truth[i_video].x, human_truth[i_video].y, person_radius*np.cos(theta), person_radius*np.sin(theta), width=0.035, color="red")

            # This is for plotting the trajectory
            # ax.add_artist(plt.Circle((robot.x, robot.y), robot.radius/8, fill=True, color="blue"))
            # ax.set_xlim(-4, 4)
            # ax.set_ylim(-4, 4)
            # ax.axis('equal')
            ax.title.set_text(r'Scenario {0}. Time: {1:.2f} s'.format(scenario_i, time))
        else:
            robot_ours = plt.Circle((ours_traj[0].x, ours_traj[0].y), person_radius, fill=True, color="green")
            robot_baseline = plt.Circle((baseline_traj[0].x, baseline_traj[0].y), person_radius, fill=True, color="red")
            ax.add_artist(robot_ours)
            ax.add_artist(robot_baseline)
            ax.add_artist(goal_artist)
            ax.legend([robot_artist, obstacle, robot_baseline, robot_ours, goal_artist], ['Robot', 'Obstacle', 'Goal', 'Baseline', 'Ours'], loc='lower right', markerscale=0.4)
    else:
        print(batch_id, scenario_i)
        scenario_i = scenario_i + 1
        if scenario_i > 127:
            batch_id = batch_id+1
            scenario_i = -1
        if batch_id == 5:
            exit(0)
        scenario_info = read_scenario(batch_id, scenario_i, "good_scenarios")
        while scenario_info is None:
            scenario_i = scenario_i+1
            if scenario_i > 127:
                batch_id = batch_id+1
                scenario_i = -1
            if batch_id == 5:
                exit(0)
            scenario_info = read_scenario(batch_id, scenario_i, "good_scenarios")
        obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj = scenario_info
        obstacles_artist, obstacles_arrow_artist, robot_artist, robot_arrow_artist, time = init_plot(ax, scenario_i, person_radius, obstacles_previous, goals, human_previous)

if __name__ == '__main__':
    anim = animation.FuncAnimation(fig, update, interval=0.8 * 1000, frames=(8+13)*100)
    anim.running = True
    ffmpeg_writer = animation.FFMpegWriter(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
    anim.save("good_scenarios.mp4", writer=ffmpeg_writer)

# plt.show()
