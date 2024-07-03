import copy, json
from scenario import Scenario
from agent import Agent
import pandas as pd
import numpy as np
np.random.seed(42)

def create_agents(opt, sce, scenario):
	agents = []   
	for i in range(opt.nagents):
		agents.append(Agent(opt, sce, scenario, index=i)) # Initializing UEs as independent agent
	return agents
    
def run_episodes(opt, sce, agents, scenario): 
    
    global_step = 0
    nepisode = 0
    action = np.zeros(opt.nagents, dtype=int)
    reward = np.zeros(opt.nagents)
    QoS = np.zeros(opt.nagents)
    state_target = np.ones(opt.nagents)  

    #Randomly generate mobile users
    mob_n = int(opt.nagents * opt.mob_p/100)
    rand_m = np.random.choice(np.arange(opt.nagents), size=mob_n, replace=False)    
    #print("Mobile UEs:",np.sort(rand_m))


    #intialize the excel file to store result
    df = pd.DataFrame(index=range(4*opt.nagents+3), columns=range((opt.nagents*2)+3+(opt.nagents*2)))
    if opt.mob == 1:
        name_ex= 'mob_'
    else:
        name_ex= 'sta_'
    name_ex = 'ql_' + name_ex + 'ue_'+str(opt.nagents)+'_mbs_'+str(sce.nMBS)+'_pbs_'+str(sce.nPBS)+'_fbs_'+str(sce.nFBS)+'_chn_'+str(sce.nChannel)+'.xlsx'
    path_ex = r'Result\\'+name_ex
    df.to_excel(path_ex, index=True)



    while nepisode < opt.nepisodes:
        state = np.zeros(opt.nagents, dtype=int)  # Reset the state   
        next_state = np.zeros(opt.nagents, dtype=int)  # Reset the next_state
        nstep = 0
        rate_l = np.zeros(opt.nagents)
        data = dict()
        while nstep < opt.nsteps:            
            eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-opt.eps_decay*nepisode)
            locx_d = []
            locy_d = []
            
            #Store current location of UEs in excel file 
            for i in range(opt.nagents):
                curr_loc = agents[i].Get_Location()
                locx_d.append(curr_loc[0])
                locy_d.append(curr_loc[1])


            #select action    
            for i in range(opt.nagents):
                action[i] = agents[i].Select_Action(state, scenario, eps_threshold)  # Select action
                #print(i,action[i])


            #obtain reward
            for i in range(opt.nagents):
                #print(i,action[i])
                QoS[i], reward[i], rate = agents[i].Get_Reward(action, action[i], state, scenario)  # Obtain reward and next state
                next_state[i] = QoS[i]
                rate_l[i] = rate
                #print(rate)


            state = copy.deepcopy(next_state)  
            if np.all(state == 1):  # If QoS threshold is satisfied, break
                break


            #Update mobility
            if opt.mob == 1 and nstep % 10 == 0:
                for i in rand_m:
                    agents[i].mob_update()
                    #print(agents[i].Get_Location())
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
    BS = scenario.Get_BaseStations()
    agents = create_agents(opt, sce, scenario)  
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















