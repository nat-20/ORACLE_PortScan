ONOS_VERSION = 2.1.0
ONOS_MD5 = 6ca21242cf837a726cfbcc637107026b
ONOS_URL = http://repo1.maven.org/maven2/org/onosproject/onos-releases/$(ONOS_VERSION)/onos-$(ONOS_VERSION).tar.gz
ONOS_TAR_PATH = ~/onos.tar.gz
APP_OAR = app/target/template-1.0-SNAPSHOT.oar
OCI = 127.0.0.1 #IP addres of the site where is executed the ONOS controller.


p4:
	cd p4src && make build

onos-cli:
	onos


app-build: 
	$(info ************ BUILDING ONOS APP ************)
	-rm -rf app/target	
	-cd app && mvn clean package


$(APP_OAR):
	$(error Missing app binary, run 'make app-build' first)

app-reload: $(APP_OAR)
	$(info ************ RELOADING ONOS APP ************)
	/opt/onos/bin/onos-app $(OCI) reinstall! app/target/template-1.0-SNAPSHOT.oar

test-all:
	$(info ************ RUNNING ALL PTF TESTS ************)
	cd ptf && make all

reset:
	-cd ~ && ./kill_onos.sh
	-cd p4src && make clean
	-cd ptf && make clean
	-sudo rm -rf app/target
	-sudo mn -c
	-sudo rm -rf /tmp/bmv2-*

move: 
	-rm -rf app/src/main/resources/*
	-cp p4src/build/* app/src/main/resources/	


fast:	move
	-sudo rm -rf app/target
	-make app-build
	-make app-reload 

log:	
	sudo tail -f /tmp/bmv2-s1-log
