# ORACLE_ddos

ORACLE: c**O**llabo**R**ation of d**A**ta and **C**ontrol p**L**an**E**s to detect DDoS attacks in a Software-Defined Networking (SDN) architecture. This DDoS detection system is composed by two modules: a control plane implementation developed in an **ONOS** controller, and a data plane implementation developed using the **P4 lenguage**.  In order to communicate both planes is used the **P4Runtime** interface which allows controlling in real-time the data plane elements of a p4 device. For more information about ORACLE, read and cite the original [paper](https://dl.ifip.org/db/conf/im/im2021-ws4-dissect/213242.pdf).

The following image shows the operating architecture of the detection system. On the right side of the image we can see the repository folders distribution where you can associate the folder color with a specific part of the architecute.  

![](https://github.com/sebitas0623/ORACLE_ddos/blob/master/images/Archit.png)


## Getting Started

These instructions will guide you to run the detection mechanism under an ONOS+P4 experimental scenario. We recommend using an Ubuntu 18.04 virtual machine over a host machine with at least 12 GB of RAM and core i5 or ryzen 5 processor.

### Prerequisites

To run the DDoS detection system is needed to install at the virtual machine the following SDN controller:

- [ONOS (2.2.0 or greater)](https://wiki.onosproject.org/display/ONOS/Development+Environment+Setup "ONOS")

Builds and installs the following repositories needed for developing and testing P4 support in:

- [BMv2](https://github.com/p4lang/behavioral-model) (P4 software switch)
- [PI](https://github.com/p4lang/PI)
- [p4c](https://github.com/p4lang/p4c) (P4 compiler)

The best way to install the previus repositories is executing the tool that brings **ONOS**:

```
bash $ONOS_ROOT/tools/dev/p4vm/install-p4-tools.sh
```

##### Note:
It is possible that in the middle of the tool execution could appear the next issue: "sudo: pip2.7: command not found". The solution is to open the file ($ONOS_ROOT/tools/dev/p4vm/install-p4-tools.sh) and change all "pip2.7" by "pip". Then, run the file again.

## Preparing the environment

![](https://github.com/sebitas0623/ORACLE_ddos/blob/master/images/DESappP4GitHub.png)

1. Run the ONOS controller activating only the APPs required by the implementation. These applications are the bmv2-driver, gui, and the custom application in charge of the extraction of the flow information and the features calculation. Although, the last one mentioned is not installed yet into the ONOS, at the moment to be done it, the application is activivated automaticly. To run the ONOS controller, you must be located in **~/onos** directory. Then, execute the next command on a terminal (**Terminal #1**):  
```
ONOS_APPS=drivers.bmv2,gui,org.p4.template ok clean
```
On another terminal (**Terminal #2**), download **ORACLE_ddos** repository into the home **(~/)** of you virtual machine. In the root of this new directory you can find a **Makefile** which containts the different commands needed to execute each part of the proyect in a easy way.      

2. Get into the ORACLE ddos folder and compile the p4 application with the follow Make command:
```
make p4
```
If the compilation was successful, the p4 compilator crates a new folder (**./p4src/build**) with two files (**bmv2.json** and **p4info.txt**). Make sure that the compilation does not show any error.

3. The next step is to package together all the project as an ONOS application (.oar). This package, named **org.p4.template.oar**, will contain the Pepilene, the custom module, and the P4 application already compilated. After that, it must be installed in ONOS using the interface provided. To do that, you just need execute the follow make command (the process can be watched in real-time in the ONOS logs or at the terminal #2):
```
make fast
```

4. Creat the mininet topology, It will be conformed by a only BMv2 switch and two hosts diretly connected. For creating the topology, execute the next make command being into **/topoP4** folder.
```
make topo
```

5. On a new terminal (**Terminal #3**) and being located into the **/topoP4** folder, send the bmv2-s1-netcfg.json file to ONOS controller. This file is created at the moment that the topology is launched and contents the switch interface information.
```
make netcfg
```
Once the netcfg file arrive to the controller, a communication channel  is stablished between ONOS  and the switch. Then, the P4 application is installed immediately at the switch (This can be watched at the ONOS logs or at the terminal #2).

6. Install the CLONE SESSION ID to be possible cloning packets at the switch. Execute the next Make command being located also into the /topoP4 folder.
```
make mirror
```

## Running

1. The first step to run the a test is executing the flows classification service. Go the ML_model folder where you will find a python script named **API_REST_Clasificador.py**. This script loads the classification model (RF or KNN) and stays waiting for flows classification requests sent by the ONOS controler. This also shows at the terminal the classification result for each group of flows, and the accuracy score aggregated. The classification service is execute as follow: 
```
python3 API_REST_Clasificador.py 
```

2. Download into the virtual machine the wordload from [here](https://drive.google.com/drive/folders/1UU23vmK1P-I9YjN7MXBa_MVSNNVBhfA1?usp=sharing). It is a .pcap file with 45 minutes of traffic where 10 minutes are of DDoS attack. On the other hand, all packets of this workliad were previously marked (00: Benign, 11: DDoS) modifying the last two bits of the IPv4 ToS header field. This allows to compare the real flow tag with the given by the classificator.

The workload is a portion of  the traffic generated in the creation of the [CIC-IDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html).

3. On the Terminal #2 must be active the mininet topology. Please open the terminal of the host **H1** putting the command: **xterm h1**. Then, change the network interface mtu value using the next command on the H1 terminal (**Terminal #4**):
```
ifconfig h1-eth0 mtu 12000
```

4. H1 host will reproduce the workload through **tcpreplay**. Execute the next command on Terminal #4. you must be located at the folder that contains the workload:
```
sudo tcpreplay -i h1-eth0 R_Test_pares.pcap
```
