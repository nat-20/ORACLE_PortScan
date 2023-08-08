import subprocess
import sys


def update(opt):
    thrift_port = 60004
    arg = ['simple_switch_CLI', '--thrift-port', '%d'%(thrift_port) ]
    print(arg)
    
    if opt == "reset":
        print("RESET_REGISTERS")
        
        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.FlowState"
        print(command)   
        out, error = p.communicate(input=command)

        """p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.IAT_MAX"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.IAT_MIN"
        print(command)
        out, error = p.communicate(input=command)"""

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.InitTimeFlow"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.LastTimePacket"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.PktLenMax"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.PktLenMin"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.NumPacketsTCP"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.NumPacketsUDP"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.TotLenPkts"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.TotPkts"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.indexsFWD0"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.indexsBWD0"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.indexsFWD1"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.indexsBWD1"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.ContIndexs"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.InitTimeWindow"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.WindowId"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.Carril"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.colitions"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.TotLenSquare"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.TotIAT"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.TotIATsquare"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.tag"
        print(command)
        out, error = p.communicate(input=command)

        """p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_ingress.test"
        print(command)
        out, error = p.communicate(input=command)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_reset c_egress.test2"
        print(command)
        out, error = p.communicate(input=command)"""


    if opt == "show":
        print("SHOW_REGISTERS")

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.FlowState"
        out, error = p.communicate(input=command)
        tramo(out)

        """p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.IAT_MAX"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.IAT_MIN"
        out, error = p.communicate(input=command)
        tramo(out)"""

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.InitTimeFlow"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.LastTimePacket"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.PktLenMax"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.PktLenMin"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.TotLenPkts"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.TotPkts"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.NumPacketsTCP"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.NumPacketsUDP"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.indexsFWD0"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.indexsBWD0"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.indexsFWD1"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.indexsBWD1"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.ContIndexs"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.InitTimeWindow"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.WindowId"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.Carril"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.colitions"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.TotLenSquare"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.TotIAT"
        out, error = p.communicate(input=command)
        tramo(out)


        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.TotIATsquare"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.tag"
        out, error = p.communicate(input=command)
        tramo(out)

        """p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_ingress.test"
        out, error = p.communicate(input=command)
        tramo(out)

        p = subprocess.Popen(arg, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = "register_read c_egress.test2"
        out, error = p.communicate(input=command)
        tramo(out)"""


def tramo(text):
    t = text.split('RuntimeCmd')
    t2 = t[1].split('\n')
    t3 = t2[0].split('=')
    print(t3[0]+'\t'+t3[1])
    #print(t2)

#Begin.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Parameters: --option[reset,show]")
        sys.exit(1)

    update(sys.argv[1])
