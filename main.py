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
    parser.add_argument('--plot_dqn', type=int, help='1: learning curve 2: hyperparamters 3: improvement')
    parser.add_argument('--plot_pg', type=int, help='1: learning curve 3: improvement')

    parser.add_argument('--env-name', type=str, default='CartPole-v0')
    parser.add_argument('--max-steps', type=int, default=200, metavar='N')
    parser.add_argument('--num-episodes', type=int, default=1000, metavar='N')
    parser.add_argument('--num-trajs', type=int, default=10, metavar='N')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G')
    parser.add_argument('--hidden_layer', type=int, default=128, metavar='N')
    parser.add_argument('--seed', type=int, default=777, metavar='N',)
    parser.add_argument('--reinforce', action ='store_true', help='Use REINFORCE instead of importance sampling')

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
        ret = None
        while(ret==None or ret[-1] < 25):
            window_size = 20
            ret = next(dqn_training)
            plt.plot(ret)
            plt.xlabel('number of episodes playing')
            plt.ylabel('average reward of last {} episodes'.format(window_size))
            plt.title('learning curve of dqn with pacman')
            plt.savefig('dqn-learning_curve1.png')
            plt.close()

    elif args.plot_dqn == 2:
        import matplotlib.pyplot as plt
        game_name = 'MsPacmanNoFrameskip-v0'
        #game_name = 'AtlantisNoFrameskip-v0'
        #game_name = 'UpNDownNoFrameskip-v0'
        #game_name = 'AmidarNoFrameskip-v0'
        #game_name = 'GopherNoFrameskip-v0'
        #game_name = 'SkiingNoFrameskip-v0'
        env_name = args.env_name or game_name
        env1 = Environment(env_name, args, atari_wrapper=True)
        env2 = Environment(env_name, args, atari_wrapper=True)
        env3 = Environment(env_name, args, atari_wrapper=True)
        env4 = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn_plot import AgentDQN
        agent1 = AgentDQN(env1, args, target_update_freq=100000)
        agent2 = AgentDQN(env2, args, target_update_freq=10000)
        agent3 = AgentDQN(env3, args, target_update_freq=1000)
        agent4 = AgentDQN(env4, args, target_update_freq=1)

        dqn_training1 = agent1.train()
        dqn_training2 = agent2.train()
        dqn_training3 = agent3.train()
        dqn_training4 = agent4.train()

        for episode in range(6000):
            window_size = 20
            ret1 = next(dqn_training1)
            ret2 = next(dqn_training2)
            ret3 = next(dqn_training3)
            ret4 = next(dqn_training4)
            if episode % 10 == 0:
                plt.plot(ret1, label='target_update_freq=100000')
                plt.plot(ret2, label='target_update_freq=10000')
                plt.plot(ret3, label='target_update_freq=1000')
                plt.plot(ret4, label='target_update_freq=1')
                plt.xlabel('number of episodes playing')
                plt.ylabel('average reward of last {} episodes'.format(window_size))
                plt.legend()
                plt.title('learning curve of dqn with {}'.format(game_name))
                plt.savefig('dqn-learning_curve2.png')
                plt.close()
                #print(ret)

    elif args.plot_dqn == 3:
        print('Duel vs DQN')
        #print('DDQN vs DQN')
        import matplotlib.pyplot as plt
        #game_name = 'MsPacmanNoFrameskip-v0'
        #game_name = 'AlienNoFrameskip-v4'
        game_name = 'AtlantisNoFrameskip-v4'
        #game_name = 'UpNDownNoFrameskip-v0'
        #game_name = 'AsterixNoFrameskip-v0'
        #env_name = args.env_name or game_name
        env_name = game_name
        env1 = Environment(env_name, args, atari_wrapper=True)
        env2 = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn_plot import AgentDQN
        from Duel import AgentDuel
        from DDQN import AgentDDQN
        agent1 = AgentDQN(env1, args)
        agent2 = AgentDuel(env2, args)
        #agent2 = AgentDDQN(env2, args)

        dqn_training1 = agent1.train()
        dqn_training2 = agent2.train()
        for episode in range(3500):
            window_size = 20
            ret1 = next(dqn_training1)
            ret2 = next(dqn_training2)
            if episode % 10 == 0:
                plt.plot(ret1, label='DQN')
                #plt.plot(ret2, label='DDQN')
                plt.plot(ret2, label='Duel')
                plt.xlabel('number of episodes playing')
                plt.ylabel('average reward of last {} episodes'.format(window_size))
                plt.legend()
                plt.title('learning curve of dqn with {}'.format(env_name))
                #plt.savefig('dqn-learning_curve_ddqn.png')
                plt.savefig(game_name + 'dqn-learning_curve_duel.png')
                plt.close()
                #print(ret)

    if args.plot_pg == 1:
        import matplotlib.pyplot as plt
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent_pg_plot import AgentPG
        agent = AgentPG(env, args)
        print(agent.model)

        pg_training = agent.train()

        for _ in range(600):
            result = next(pg_training)
            """plot"""
            plt.plot(range(len(result)), result, label='policy gradient')
            plt.title('learning curve of pg with {}'.format(env_name))
            plt.ylabel('average reward of last 20 episodes')
            plt.xlabel('number of episodes playing')
            plt.legend()
            plt.savefig('pg-learning_curve.png')
            plt.close()
    
    if args.plot_pg == 3:
        import matplotlib.pyplot as plt
        env_name = args.env_name or 'LunarLander-v2'
        env1 = Environment(env_name, args, atari_wrapper=False)
        env2 = Environment(env_name, args, atari_wrapper=False)
        from agent_pg_plot import AgentPG
        from importance_sampling import Agent
        agent1 = AgentPG(env1, args)
        agent2 = Agent(args, env2.observation_space.shape[0], env2.action_space, env2)

        pg_training1 = agent1.train()
        pg_training2 = agent2.train()

        for _ in range(600):
            result1 = next(pg_training1)
            result2 = next(pg_training2)
            """plot"""
            plt.plot(range(len(result1)), result1, label='online-policy gradient')
            plt.plot(range(len(result2)), result2, label='offline-policy gradient by importance sampling')
            #plt.plot(result2)
            plt.title('learning curve of pg with {}'.format(env_name))
            plt.ylabel('average reward of last 20 episodes')
            plt.xlabel('number of episodes playing')
            plt.legend()
            #plt.grid(True)
            plt.savefig('reward-episodes2.png')
            plt.close()
        

        #env2.close()

if __name__ == '__main__':
    args = parse()
    run(args)
