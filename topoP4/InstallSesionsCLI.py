#-*- coding: UTF-8 -*-
#Set Switch_CLI

#-----------------------------------------------------
#-------------- Form de ejecutarse -------------------
#---------sudo python2 updateCLI.py ------------------
#-----------------------------------------------------


import subprocess


def update():
    print("UPDATE_SWITCHES")

    
    thrift_port = 60004
    
    arg = ['simple_switch_CLI', '--thrift-port', '%d'%(thrift_port) ]
    print(arg)

    #---- Ejemplo de generar un comando de consola -----
    #command = 'table_set_default limiar_time add_limiar_time %s'%(time)
    #command = command +'\n'+ 'table_add get_flag_white set_flag_white  0x00000009&&&0x0000000f 0x00000000&&&0x00000000 => 1 1'
    #command = command +'\n'+ 'table_add get_flag_white set_flag_white  0x00000000&&&0x00000000 0x00000009&&&0x0000000f => 1 1'
    #print(command)
    

    p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command = "mirroring_add 99 255"
    print(command)   
    out, error = p.communicate(input=command)
    p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command = "mirroring_add 69 2"
    print(command)
    out, error = p.communicate(input=command)
    #print(out)




#Begin.
if __name__ == '__main__':

    update()
