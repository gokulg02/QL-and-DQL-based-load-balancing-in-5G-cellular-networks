import copy, json, argparse
import torch
from scenario import Scenario
from agent import Agent
import pandas as pd
import numpy as np
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_agents(opt, sce, scenario, device):
	agents = []   # Vector of agents
	for i in range(opt.nagents):
		agents.append(Agent(opt, sce, scenario, index=i, device=device)) # Initializationa neural net for each agent
	return agents
    
def run_episodes(opt, sce, agents, scenario): 
    global_step = 0
    nepisode = 0
    action = torch.zeros(opt.nagents,dtype=int)
    reward = torch.zeros(opt.nagents)
    QoS = torch.zeros(opt.nagents)
    state_target = torch.ones(opt.nagents)  

    #Randomly generate mobile users
    mob_n = int(opt.nagents * opt.mob_p/100)
    rand_m = np.random.choice(np.arange(opt.nagents+1), size=mob_n, replace=False)    
    #print("Mobile UEs:",np.sort(rand_m))
    
    #intialize the excel file to store result
    df = pd.DataFrame(index=range(4*opt.nagents+3), columns=range((opt.nagents*2)+3+(opt.nagents*2)))
    if opt.mob == 1:
        name_ex= 'mob_'
    else:
        name_ex= 'sta_'
    name_ex = 'dql_' + name_ex + 'ue_'+str(opt.nagents)+'_mbs_'+str(sce.nMBS)+'_pbs_'+str(sce.nPBS)+'_fbs_'+str(sce.nFBS)+'_chn_'+str(sce.nChannel)+'.xlsx'
    path_ex = r'Result\\'+name_ex
    df.to_excel(path_ex, index=True)

    while nepisode < opt.nepisodes:
        state = torch.zeros(opt.nagents)  # Reset the state   
        next_state = torch.zeros(opt.nagents)  # Reset the next_state
        nstep = 0
        rate_l = torch.zeros(opt.nagents)
        data = dict()
        while nstep < opt.nsteps:            
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            locx_d = []
            locy_d = []
            for i in range(opt.nagents):
                curr_loc = agents[i].Get_Location()
                locx_d.append(curr_loc[0])
                locy_d.append(curr_loc[1])

            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max 

            for i in range(opt.nagents):
                action[i] = agents[i].Select_Action(state, scenario, eps_threshold)  # Select action
            for i in range(opt.nagents):
                QoS[i], reward[i], rate = agents[i].Get_Reward(action, action[i], state, scenario)  # Obtain reward and next state
                next_state[i] = QoS[i]
                rate_l[i] = rate
                #print(rate)
            for i in range(opt.nagents):
                agents[i].Save_Transition(state, action[i], next_state, reward[i], scenario)  # Save the state transition
                agents[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    agents[i].Target_Update()
            state = copy.deepcopy(next_state)  # State transits 
            if torch.all(state.eq(state_target)):  # If QoS is satisified, break
                break

            #Update mobility
            if opt.mob == 1 and nstep % 10 == 0:
                for i in rand_m:
                    agents[i].mob_update()
                    # print(agents[i].Get_Location())
            nstep += 1
        #save data in excel file
        ac_l = action
        ac_l = ac_l// sce.nChannel
        r_l = rate_l
        ac_l = list(ac_l)
        r_l = list(r_l)
        avg = sum(r_l) / len(r_l)
        d = [nepisode, nstep]
        d.extend(ac_l)
        d.extend(r_l)
        d.append(avg)
        d.extend(locx_d)
        d.extend(locy_d)
        print('Episode Number:', nepisode, 'Training Step:', nstep)
        df.iloc[nepisode] = d
        labels = ['nepisodes','nsteps']
        labels.extend([f"ue{i}_bs" for i in range(1, opt.nagents+1)])
        labels.extend([f"ue{i}_rate" for i in range(1, opt.nagents+1)])
        labels.extend(["avg_rate"])
        labels.extend([f"ue{i}_x" for i in range(1, opt.nagents+1)])
        labels.extend([f"ue{i}_y" for i in range(1, opt.nagents+1)])
        df = df.set_axis(labels)
        df.to_excel(path_ex, index=False)

        nepisode += 1
   
                
def run_trial(opt, sce):
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)  # Initialization 
    run_episodes(opt, sce, agents, scenario)    
        
if __name__ == '__main__':

    class DotDic(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __deepcopy__(self, memo=None):
            return DotDic(copy.deepcopy(dict(self), memo=memo))

    sce = DotDic(json.loads(open(r"Config\config_1.json").read()))
    opt = DotDic(json.loads(open(r"Config\config_2.json").read()))  

    for i in range(1):
        trial_result_path = None
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)