# QL and DQL based load balancing in 5G HetNets
 
## Summary

* Load balancing in mobile networks involves connecting mobiles or User Equipments (UEs) to Base Stations (BSs) such that the number of UEs supported by each BS remains equal.
* In 5G Hetrogenous Networks (HetNets), there are different types of BSs (namely Macro (MBS), Pico (PBS) and Femto (FBS)). MBS offers larger coverage area but slower data rates. On the other hand, PBS and FBS offer limited coverage but fast data rates. 5G HetNets also allow a same channel to be used for communication by different UEs.
* A UE must choose a BS and channel pair to connect with such that it recieves maximal data rates. This makes connecting all UEs to a BS in a HetNets such that all UEs recieve maximum data rates a NP - hard problem.
* Conventional algorithms for load balancing need a central node to coordinate the process. But, Reinforcement Learning algorithms can make the UEs act as independent agents and eliminate the need for a central node.
* In this work, load balancing in carried out using Q - Learning and Deep Q - Learning algorithms. A HetNet consisting of 1 MBS, 2 PBSs, 4 FBSs and 30 UEs are considered for simulation and this network is illustrated below.
  ![Before_LB](https://github.com/user-attachments/assets/1ad042f9-dddc-42c4-a11e-b6abb71d2794)
* After performing load balancing, UE associations obtained is depiced below and data rates achieved by each UE is obtained as an excel sheet in [Results folder](https://github.com/gokulg02/QL-and-DQL-based-load-balancing-in-5G-cellular-networks/tree/main/DQL/Result).
* 
  ![After_LB](https://github.com/user-attachments/assets/6d80f055-4e25-426d-a544-c7f692c029e4)
* Further, the performance of the proposed algorithms in the presence of UE mobility and changes to the network is analyzed. The result all the analyses carried out are presented in this [document](https://github.com/gokulg02/QL-and-DQL-based-load-balancing-in-5G-cellular-networks/blob/main/Paper.pdf).

## Organization of Repo

The folders DQL and QL contains the code for both the algorithms respectively. The files are organized in them as follows:

* Config
   * config_1.json   (Stores HetNet hyperparameters)
   * config_2.json   (Stores algorithm hyperparameters)   
* main.py   (Simulates the entire HetNet)
* agent.py   (Class for UEs)
* scenario.py   (Class for BS)
* Result   (Folder to store the output excel files)


## Built With

* Python
* PyTorch
* NumPy
* Pandas



[Next-url]: https://nextjs.org/

