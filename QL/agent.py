import numpy as np
from numpy import pi
from random import random
from scenario import Scenario, BS
import random

class Agent:  # Define the agent (UE)
    
    def __init__(self, opt, sce, scenario, index):  # Initialize the agent (UE)
        self.opt = opt
        self.sce = sce
        self.id = index
        self.location = self.Set_Location(scenario,index)
        self.q_val = np.zeros(scenario.BS_Number()*self.sce.nChannel)  # Initialize the Q-value

    def Set_Location(self, scenario,index):  # Initialize the location of the agent

        ue_l = [[791.443244459332, 406.7483781864429], [151.80899717911558, 251.48136688079475], [609.9863405563093, 664.0355601741302], [580.3556695903344, 410.2009498561434], [399.05336434539606, 125.71597725013851], [570.4582115339467, 486.5180119645425], [606.9441507727184, 943.4789835106141], [586.6310480105276, 694.8109425118948], [227.5389439980242, 457.2737352185727], [415.82170950185224, 817.6480285935022], [750.2568796804994, 800.5573453655306], [319.55107801624524, 701.1823766851991], [574.0235727830111, 170.5489312102672], [277.4686379246533, 480.0440877812888], [868.5678923290097, 610.7309846414369], [686.6261485637284, 842.1357824718555], [621.0055799054742, 459.74397091043664], [676.262442218967, 41.37209010828269], [725.6041553485736, 658.9220166710684], [115.3062756833894, 651.884808652659], [325.41034197071156, 505.29244937725747], [578.0738275020013, 449.9839252040673], [367.1317549117573, 283.1105699966573], [223.06074599678772, 464.8941797632218], [647.1355122059665, 839.158828403018], [577.4243409522898, 13.788494203146968], [882.6173022330563, 202.54624658117797], [840.967489357404, 317.7469728534535], [549.51816466793, 640.2536162845709], [451.529042383493, 594.6434809977596], [458.3111953483962, 808.9196756263526], [217.0748019029624, 856.5637524619326], [244.46293379913172, 429.7544647732094], [560.4608892115726, 322.30493220284194], [636.0567046086276, 488.7646072172186], [639.1458292429279, 916.7728764308713], [514.8550191094192, 465.94331654198567], [444.71252680943303, 83.27751288164609], [892.4394484162597, 696.999659893687], [723.4689261534023, 699.1935520540596], [168.04355195557434, 175.0680485538134], [764.9872195615229, 611.8314789413764], [373.1282292159829, 748.2924368704037], [222.74780950548478, 175.14192721973336], [36.198832849110374, 581.8025854943118], [460.42174114590426, 331.6785132234936], [95.81195313418806, 336.2065785050431], [61.309867548786485, 517.1103091315433], [175.32086467022407, 658.9549194803594], [562.0907098620031, 549.99525412333], [441.9719235683466, 432.99224845343815], [220.06909629448586, 484.9107846560583], [502.11826522266927, 976.3267066142414], [511.1696231350719, 179.8891281478745], [711.7297291252333, 611.2207857257848], [642.4659316051586, 728.344742791062], [672.1682981820295, 49.68623140442725], [775.0547232324557, 212.43115892864364], [673.9503116104808, 913.1093176332329], [41.98228462202485, 384.41660144106095], [856.8872784040171, 227.06867291498173], [717.1944833182262, 679.7982902006988], [285.88879528999, 605.5470636877864], [789.8528571548802, 152.88472710095863], [458.4052965477597, 497.18694516266277], [556.3234243930549, 818.0887024900987], [409.44459040807084, 647.5332492255852], [284.4781438057567, 935.0606342577063], [395.2285572455319, 155.44144152410212], [796.7820235174494, 446.82264474692573], [494.50700677444024, 990.4912341326312], [389.1981820355833, 834.7164226757877], [759.7171331817347, 561.2954674199327], [109.68229785777169, 493.4292398168092], [479.69105302123506, 611.6119818169674], [531.2296293446475, 975.4904644077096], [310.092482117755, 512.6036770273838], [524.7690880045714, 995.7813084486667], [529.9008940950895, 91.17262397952862], [466.7431395518382, 258.5390979242573], [295.65189183096527, 275.9740330507426], [112.03874817732492, 411.29546361991095], [576.7246843918566, 370.8266075440764], [609.9824429834426, 760.9576181649575], [415.05790687285065, 445.4232081936867], [909.3381174954034, 542.8181740139619], [552.6490397173477, 853.9086650417893], [683.7516540931513, 857.1113652915478], [185.26560859222212, 771.4345176848224], [814.2351099707895, 868.0200984264711], [720.8777615949268, 690.9963407905484], [844.9407719749531, 165.0533349935186], [363.9107747864298, 285.6052013001271], [74.91380224100101, 346.35341030435745], [518.6204596708325, 863.4087548073774], [621.8390836686008, 408.179906108355], [182.01642356484876, 147.87649977835486], [330.0491414647827, 736.3771423918095], [840.0503936329065, 243.37839236014207], [587.8810405906116, 37.346210681258356]]

        Loc_agent = np.array(ue_l[index])
        return Loc_agent
    
    def mob_update(self):
        current_loc = self.location
        step_size = 10
        center = [500, 500]
        radius = 500
        angle = np.random.uniform(0, 2*np.pi)
        x_step = step_size * np.cos(angle)
        y_step = step_size * np.sin(angle)
        
        # Update position while ensuring it stays within the circle
        new_position = [current_loc[0] + x_step, current_loc[1] + y_step]
        distance_to_center = np.sqrt((new_position[0] - center[0])**2 + (new_position[1] - center[1])**2)
        if distance_to_center > radius:
            # Reflect the step if it goes beyond the circle boundary
            reflected_position = [center[0] + (radius / distance_to_center) * (new_position[0] - center[0]),
                                  center[1] + (radius / distance_to_center) * (new_position[1] - center[1])]
            new_position = reflected_position
        
        self.location = new_position

        
    
    def Get_Location(self):
        return self.location
     
    def Select_Action(self, state, scenario, eps_threshold):   # Select action for a user based on the network state
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels    
        sample = random.random()
        if sample < eps_threshold:  # epsilon-greeedy policy
            action = np.argmax(self.q_val) 
        else:           
            action = random.randint(0, scenario.BS_Number()*self.sce.nChannel-1)  # exploration
        return action      
        		
    def Get_Reward(self, action, action_i, state, scenario):  # Get reward for the state-action pair
        BS = scenario.Get_BaseStations()
        L = scenario.BS_Number()  # The total number of BSs
        K = self.sce.nChannel  # The total number of channels 

        BS_selected = int(action_i) // K
        Ch_selected = int(action_i) % K  # Translate to the selected BS and channel based on the selected action index
        Loc_diff = BS[BS_selected].Get_Location() - self.location
        distance = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))  # Calculate the distance between BS and UE
        Rx_power = BS[int(BS_selected)].Receive_Power(distance)  # Calculate the received power
        
        Rate = 0

        if Rx_power == 0.0:
            reward = self.sce.negative_cost  # Out of range of the selected BS, thus obtain a negative reward
            QoS = 0  # Definitely, QoS cannot be satisfied
        else:                    # If inside the coverage, then we will calculate the reward value
            Interference = 0.0
            for i in range(self.opt.nagents):   # Obtain interference on the same channel
                BS_select_i = action[i] // K
                Ch_select_i = action[i] % K   # The choice of other users
                if Ch_select_i == Ch_selected:  # Calculate the interference on the same channel
                    Loc_diff_i = BS[int(BS_select_i)].Get_Location() - self.location
                    distance_i = np.sqrt((Loc_diff_i[0]**2 + Loc_diff_i[1]**2))
                    Rx_power_i = BS[int(BS_select_i)].Receive_Power(distance_i)
                    Interference += Rx_power_i   # Sum all the interference
            Interference -= Rx_power  # Remove the received power from interference
            Noise = 10**((self.sce.N0)/10)*self.sce.BW  # Calculate the noise
            SINR = Rx_power/(Interference + Noise)  # Calculate the SINR      
            if SINR >= 10**(self.sce.QoS_thr/10):
                QoS = 1
                reward = 1
            else:
                QoS = 0   
                reward = self.sce.negative_cost
            Rate = self.sce.BW * np.log2(1 + SINR) / (10**6)      # Calculate the rate of UE 

        self.q_val[action_i] = self.q_val[action_i] + self.opt.alpha * (reward + self.opt.gamma*np.max(self.q_val) - self.q_val[action_i])  # Update the Q-value
        return QoS, reward, Rate
    