"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--plot_dqn', type=int, help='whether plot DQN')
    parser.add_argument('--plot_pg', action='store_true', help='whether plot policy gradient')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.train()

    if args.train_dqn:
        env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.train()

    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=30)

    if args.test_dqn:
        env = Environment('MsPacmanNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=50)
    
    if args.plot_dqn == 1:
        import matplotlib.pyplot as plt
        env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn_plot import AgentDQN
        agent = AgentDQN(env, args)

        dqn_training = agent.train()
        for _ in range(50):
            window_size = 20
            ret = next(dqn_training)
            plt.plot(ret)
            plt.xlabel('number of episodes playing')
            plt.ylabel('average reward of last {} episodes'.format(window_size))
            plt.title('learning curve of dqn with pacman')
            plt.savefig('dqn-learning_curve.png')

    elif args.plot_dqn == 2:
        import matplotlib.pyplot as plt
        env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn_plot import AgentDQN
        agent1 = AgentDQN(env, args, target_update_freq=1000)
        agent2 = AgentDQN(env, args, target_update_freq=900)
        agent3 = AgentDQN(env, args, target_update_freq=800)
        agent4 = AgentDQN(env, args, target_update_freq=700)

        dqn_training1 = agent1.train()
        dqn_training2 = agent2.train()
        dqn_training3 = agent3.train()
        dqn_training4 = agent4.train()

        for _ in range(50):
            window_size = 20
            ret1 = next(dqn_training1)
            ret2 = next(dqn_training2)
            ret3 = next(dqn_training3)
            ret4 = next(dqn_training4)
            plt.plot(ret1)
            plt.plot(ret2)
            plt.plot(ret3)
            plt.plot(ret4)
            plt.xlabel('number of episodes playing')
            plt.ylabel('average reward of last {} episodes'.format(window_size))
            plt.title('learning curve of dqn with pacman')
            plt.savefig('dqn-learning_curve.png')
            #print(ret)
    
    if args.plot_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent_pg_plot import AgentPG
        agent = AgentPG(env, args)
        agent.train()

if __name__ == '__main__':
    args = parse()
    run(args)
