OCI = 127.0.0.1

onos-cli:
	onos

reset:
	-sudo mn -c
	-sudo rm -rf /tmp/bmv2-*

topo: reset
	$(info ************ STARTING MININET TOPOLOGY ************)
	#-sudo rm -rf bmv2-*	
	#-sudo mn -c	
	-sudo -E python2 topo.py --onos-ip $(OCI)

netcfg:
	$(info ************ PUSHING NETCFG TO ONOS ************)
	-onos-netcfg ${OCI} bmv2-s1-netcfg.json 
	#-onos-netcfg ${OCI} bmv2-s2-netcfg.json
	#-onos-netcfg ${OCI} bmv2-s3-netcfg.json


mirror:
	$(info ************ CREATING CLONE SESSION ID ************)
	-sudo python2 InstallSesionsCLI.py
