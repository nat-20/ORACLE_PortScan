/*
 * Copyright 2019-present Open Networking Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Obtencion de estadisticas para el calculo de caracteristicas del DataSet %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CICIDS-2017 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% UdeA/Colombia - UFRGS/Brasil %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% By: Sebastian Gomez Macias %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Asesor: Juan Felipe Botero Vega %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% CoAsesor: Luciano Paschoal Gaspary %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#include <core.p4>
#include <v1model.p4>

#include "include/header.p4"
#include "include/parser.p4"
#include "include/checksum.p4"

#define CPU_CLONE_SESSION_ID 99
//Lo uso para pruebas, mirar que estoy enviando al controlador, enviandolo por el puerto 2 del sw y capturarlo en wireshark
#define PORT_CLONE_SESSION_ID 69
#define CPU_PORT 255

#define WindowDuration 40000000 //20 segundos en microsegundos 
#define NumFlows 800000
//#define NumFlows 20

#define FIN 1           //000001
#define SYN 2           //000010
#define RST 4           //000100
#define PSH 8           //001000
#define ACK 16          //010000
#define FIN_ACK 17      //010001
#define SYN_ACK 18      //010010
#define RST_ACK 20      //010100
#define PSH_ACK 24      //011000
#define URG 32          //100000
#define URG_ACK 48      //110000
#define ECE 64          //1000000
#define CWR 128         //10000000

//Valores que puede tener la metadata instance_type y 
//sus respectivas interpretaciones
const bit<32> NORMAL = 0;
const bit<32> CLONE1 = 1; //Paquete clonado de ingress a egress
const bit<32> CLONE2 = 2; //Paquete clonado de egress a egress
const bit<32> RECIRCULATED = 4; 
const bit<32> RESUBMIT = 6;
const bit<16> CustomEtherType = 0x6969; //En DEC: 26985
const bit<2> C_Derecho = 0;
const bit<2> C_Izquierdo = 1;

const bit<2> DDoS_Tag = 3;
const bit<2> Benign_Tag = 0; 

//------------------------------------------------------------------------------
// INGRESS PIPELINE
//------------------------------------------------------------------------------

control c_ingress(inout headers_t hdr,
                    inout metadata_t meta,
                    inout standard_metadata_t standard_metadata) {

    //----------------- Define registers -------------------------------------- 

    //--- Contadores de tipos de paquetes que llegan al switch
    register <bit<32>> (1) NumPacketsUDP;
    register <bit<32>> (1) NumPacketsTCP;
    //--- Timestamp del primer paquete del flujo (milisegundos)
    register <bit<48>> (NumFlows) InitTimeFlow;
    //--- Timestamp del ultimo paquete que llego al switch (milisegundos)
    register <bit<48>> (NumFlows) LastTimePacket;
    //--- Estado del flujo (0,1 o 2)
    register <bit<2>> (NumFlows) FlowState;
    //--- Estadisticas por bytes y por paquetes
    register <bit<32>> (NumFlows) TotPkts;
    register <bit<32>> (NumFlows) TotLenPkts;
    register <bit<32>> (NumFlows) PktLenMin;
    register <bit<32>> (NumFlows) PktLenMax;
    register <bit<40>> (NumFlows) TotLenSquare;
    register <bit<48>> (NumFlows) TotIAT;
    register <bit<56>> (NumFlows) TotIATsquare;
    //--- Registradores que me maneja la ventada de recoleccion de flujos y el envio de estos
    register <bit<32>> (NumFlows) indexsFWD0;
    register <bit<32>> (NumFlows) indexsBWD0;
    register <bit<32>> (NumFlows) indexsFWD1;
    register <bit<32>> (NumFlows) indexsBWD1;
    register <bit<32>> (2) ContIndexs;
    register <bit<48>> (1) InitTimeWindow;
    register <bit<16>> (1) WindowId;
    register <bit<2>> (1) Carril;
    register <bit<16>> (1) colitions;
    //--- etiqueta del flujo 
    register <bit<1>> (NumFlows) tag;
    //register <bit<48>> (1) test;

    //------------------------------- Mis Actions --------------------------------------------

    action calculate_hash() {
        hash(meta.index, 
            HashAlgorithm.crc32,
            (bit<32>) 0,
            {hdr.ipv4.src_addr,hdr.ipv4.dst_addr,meta.srcP,meta.dstP,hdr.ipv4.protocol}, 
            (bit<32>) NumFlows - 1
        );
        hash(meta.index2, 
            HashAlgorithm.crc32, 
            (bit<32>) 0, 
            {hdr.ipv4.dst_addr,hdr.ipv4.src_addr,meta.dstP,meta.srcP,hdr.ipv4.protocol}, 
            (bit<32>) NumFlows - 1
        );
    }

    action SaveIntoMetas() {
        //--- Estadisticas en sentido FWD
        InitTimeFlow.read(meta.InitTimeFlowM, meta.indF);
        LastTimePacket.read(meta.LastTimePacketM, meta.indF);
        TotPkts.read(meta.TotPktsM, meta.indF);
        TotLenPkts.read(meta.TotLenPktsM, meta.indF);
        //--- Estadisticas en sentido BWD
        InitTimeFlow.read(meta.InitTimeFlowM2, meta.indB);
        LastTimePacket.read(meta.LastTimePacketM2, meta.indB);
        TotPkts.read(meta.TotPktsM2, meta.indB);
        TotLenPkts.read(meta.TotLenPktsM2, meta.indB);
        TotLenSquare.read(meta.TotLenSquareM2, meta.indB);
        //--- Estadisticas independiente de la direccion
        TotIAT.read(meta.TotIATM, meta.indF);
        TotIATsquare.read(meta.TotIATsquareM, meta.indF);
        tag.read(meta.tagM,meta.indF);
        //--- Informacion de control
        WindowId.read(meta.WindowNumM, 0);

        if (meta.LastTimePacketM > meta.LastTimePacketM2) {
            meta.FlowDurationM = meta.LastTimePacketM - meta.InitTimeFlowM;
        } else {
            meta.FlowDurationM = meta.LastTimePacketM2 - meta.InitTimeFlowM;
        }
    }

    action zerar(){
        //--- se retorna a cero los registradores del subflujo FWD y BWD.
        FlowState.write(meta.indF, 0);
        FlowState.write(meta.indB, 0);
        InitTimeFlow.write(meta.indF, 0);
        InitTimeFlow.write(meta.indB, 0);
        tag.write(meta.indF,0);
        LastTimePacket.write(meta.indF, 0);
        LastTimePacket.write(meta.indB, 0);
        /*TotPkts.write(meta.indF, 0);
        TotPkts.write(meta.indB, 0);
        TotLenPkts.write(meta.indF, 0);
        TotLenPkts.write(meta.indB, 0);
        PktLenMin.write(meta.indF, 0);
        PktLenMin.write(meta.indB, 0);
        PktLenMax.write(meta.indF, 0);
        PktLenMax.write(meta.indB, 0);*/
    }

    action GetindexsFWD1(bit<32> conter){
        // Se obtiene el index FWD y BWD del flujo que esta siendo apuntado por "conter" del carril Izuquierdo. 
        indexsFWD1.read(meta.indF,conter - 1);
        indexsBWD1.read(meta.indB,conter - 1);
    }

    action GetindexsFWD0(bit<32> conter){
        // Se obtiene el index FWD y BWD del flujo que esta siendo apuntado por "conter" del carril Derecho.        
        indexsFWD0.read(meta.indF,conter - 1);
        indexsBWD0.read(meta.indB,conter - 1);
    }

    // funcion send para vaciar lo recolectado previamente en el carril Derecho
    action send_D(bit<32> conter){
        meta.NumFlowsByPacket = 1;
        GetindexsFWD0(conter);
        // se guarda la informacion del flujo en las metadatas
        SaveIntoMetas();
        // se concatena todas las estadisticas del flujo
        meta.Flow1 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        // se cera la informacion del flujo de los registradores de estadisticas.
        zerar();
    }

    // funcion send para vaciar lo recolectado previamente en el carril izquierdo
    action send_I(bit<32> conter){
        meta.NumFlowsByPacket = 1;
        GetindexsFWD1(conter);        
        // se guarda la informacion del flujo en las metadatas
        SaveIntoMetas();
        // se concatena todas las estadisticas del flujo
        meta.Flow1 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        // se cera la informacion del flujo de los registradores de estadisticas.
        zerar();
    }

    action send_x5_D(bit<32> conter){
        meta.NumFlowsByPacket = 5;
        GetindexsFWD0(conter);
        SaveIntoMetas();
        meta.Flow1 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-1);
        SaveIntoMetas();
        meta.Flow2 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-2);
        SaveIntoMetas();
        meta.Flow3 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-3);
        SaveIntoMetas();
        meta.Flow4 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-4);
        SaveIntoMetas();
        meta.Flow5 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
    }

    action send_x5_I(bit<32> conter){
        meta.NumFlowsByPacket = 5;
        GetindexsFWD1(conter);
        SaveIntoMetas();
        meta.Flow1 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-1);
        SaveIntoMetas();
        meta.Flow2 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-2);
        SaveIntoMetas();
        meta.Flow3 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-3);
        SaveIntoMetas();
        meta.Flow4 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-4);
        SaveIntoMetas();
        meta.Flow5 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
    }

    action send_x10_D(bit<32> conter){
        send_x5_D(conter);

        meta.NumFlowsByPacket = 10;
        GetindexsFWD0(conter-5);
        SaveIntoMetas();
        meta.Flow6 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-6);
        SaveIntoMetas();
        meta.Flow7 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-7);
        SaveIntoMetas();
        meta.Flow8 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-8);
        SaveIntoMetas();
        meta.Flow9 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD0(conter-9);
        SaveIntoMetas();
        meta.Flow10 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
    }

    action send_x10_I(bit<32> conter){
        send_x5_I(conter);

        meta.NumFlowsByPacket = 10;
        GetindexsFWD1(conter-5);
        SaveIntoMetas();
        meta.Flow6 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-6);
        SaveIntoMetas();
        meta.Flow7 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-7);
        SaveIntoMetas();
        meta.Flow8 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-8);
        SaveIntoMetas();
        meta.Flow9 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
        //----------------------
        GetindexsFWD1(conter-9);
        SaveIntoMetas();
        meta.Flow10 = meta.FlowDurationM ++ meta.TotPktsM ++ meta.TotPktsM2 ++ meta.TotLenPktsM ++ meta.TotLenPktsM2 ++ 
                    meta.TotLenSquareM2 ++ meta.TotIATM ++ meta.TotIATsquareM ++ meta.WindowNumM ++ (bit<8>)meta.tagM;
        zerar();
    }


    action clone_to_cpu() {
        clone_preserving_field_list(CloneType.I2E, CPU_CLONE_SESSION_ID, 1);
    }


    //-------------------------- Propio del Template ------------------------------------------

    action send_to_cpu() {
        standard_metadata.egress_spec = CPU_PORT;
        // Packets sent to the controller needs to be prepended with the
        // packet-in header. By setting it valid we make sure it will be
        // deparsed on the wire (see c_deparser).
        hdr.packet_in.setValid();
        hdr.packet_in.ingress_port = standard_metadata.ingress_port;
    }
    action set_out_port(bit<9> port) {
        // Specifies the output port for this packet by setting the
        // corresponding metadata.
        standard_metadata.egress_spec = port;
    }
    action _drop() {
        mark_to_drop(standard_metadata);
	//mark_to_drop();
    }

    table t_l2_fwd {
        key = {
            standard_metadata.ingress_port  : ternary;
            hdr.ethernet.dst_addr           : ternary;
            hdr.ethernet.src_addr           : ternary;
            hdr.ethernet.ether_type         : ternary;
        }
        actions = {
            set_out_port;
            send_to_cpu;
            _drop;
            NoAction;
        }
        default_action = NoAction();
    }

    //---------------------------------------------------------------------------------------

    apply {
        bit<48> InitWin;
        InitTimeWindow.read(InitWin,0);
        //--- Me inicializa el tiempo inicial de la ventana. 
        // solo se entrara una vez en todo el tiempo de vida del programa y sera en el primer paquete de todos
        if (InitWin == 0) {
            InitWin = standard_metadata.ingress_global_timestamp;
            InitTimeWindow.write(0,InitWin);
        }

        // ###################################################################################################################
        // ############################### Modulo de Manejo de carriles y ventana de tiempo ##################################
        // ###################################################################################################################
        bit<2> carril;
        Carril.read(carril,0);
        bit<32> cont; //Me guarda el numero de flujos almacenados hasta el momento en el carril necesitado.
            
        if (carril == C_Derecho)
        {
            ContIndexs.read(cont,(bit<32>)C_Izquierdo);

            // me valida si la ventana aun esta activa. si no lo está, se hace cambio de carril
            // ventana inactiva
            if (( standard_metadata.ingress_global_timestamp - InitWin) >= WindowDuration) {
                // se continua enviando las estadisticas si la ventana se acaba y aun no ha terminado de enviar las estadisticas
                // no falta nada para enviar, se procede a cambio de carril.
                if (cont == 0) {
                    //Cambio de carril
                    Carril.write(0,C_Izquierdo);
                    // se reinicia la ventana de tiempo asignando el actual timestamp 
                    InitTimeWindow.write(0,standard_metadata.ingress_global_timestamp);
                    // se incrementa en 1 el ID de la ventana
                    WindowId.read(meta.WindowNumM, 0);
                    WindowId.write(0,meta.WindowNumM + 1);
                
                // Falta estadisticas por enviar, se continuan enviando.
                } else {
                    // Nos almacenará la estadisticas del flujo en las metadas que se conservaran junto con el paquete clonado a continuacion
                    //send_I(cont);
                    //cont = cont - 1;
                    if (cont >= 10) {
                        send_x10_I(cont);
                        cont = cont - 10;
                    } else if (cont >= 5) {
                        send_x5_I(cont);
                        cont = cont - 5;
                    } else {
                        send_I(cont);
                        cont = cont - 1;
                    }
                    
                    clone_to_cpu();
                    //------------------------------------------
                    
                    ContIndexs.write((bit<32>)C_Izquierdo, cont);

                    // despues de decrementar el contador izquierdo, validamos si en la proxima ocasion habria mas flujos para enviar,
                    // caso no tenga mas flujos para enviar, se procede a cambiar de carril y actualizar la ventana.
                    if (cont == 0) {
                        //Cambio de carril
                        Carril.write(0,C_Izquierdo);
                        // se reinicia la ventana de tiempo asignando el actual timestamp 
                        InitTimeWindow.write(0,standard_metadata.ingress_global_timestamp);
                        // se incrementa en 1 el ID de la ventana
                        WindowId.read(meta.WindowNumM, 0);
                        WindowId.write(0,meta.WindowNumM + 1);
                    }

                }               

            // ventana activa
            } else {
                // si entra, hay aun estadisticas de la anterior ventana para enviar al controlador.
                if (cont != 0) {
                    //send_I(cont);
                    //cont = cont - 1;
                    if (cont >= 10) {
                        send_x10_I(cont);
                        cont = cont - 10;
                    } else if (cont >= 5) {
                        send_x5_I(cont);
                        cont = cont - 5;
                    } else {
                        send_I(cont);
                        cont = cont - 1;
                    }

                    clone_to_cpu();
                    
                    //------------------------------------------
                    // decrementamos en 1 el contador despues de clonar el paquete que sera enviado al controlador
                    ContIndexs.write((bit<32>)C_Izquierdo, cont);
                }
            }


        // carril == C_Izquierdo
        } else {
            ContIndexs.read(cont,(bit<32>)C_Derecho);

            // me valida si la ventana aun esta activa. si no lo está, se hace cambio de carril
            // ventana inactiva
            if (( standard_metadata.ingress_global_timestamp - InitWin) >= WindowDuration) {
                // se continua enviando las estadisticas así la ventana haya expirado
                // no falta nada para enviar, se procede al cambio de carril.
                if (cont == 0) { 
                    //cambio de carril
                    Carril.write(0,C_Derecho);
                    // se reinicia la ventana de tiempo asignando el actual timestamp 
                    InitTimeWindow.write(0,standard_metadata.ingress_global_timestamp);
                    // se incrementa en 1 el ID de la ventana
                    WindowId.read(meta.WindowNumM, 0);
                    WindowId.write(0,meta.WindowNumM + 1);

                // Falta estadisticas por enviar, se continuan enviando.
                } else {
                    // Nos almacenará la estadisticas del flujo en las metadas que se conservaran junto con el paquete clonado a continuacion
                    //send_D(cont);
                    //cont = cont - 1;
                    if (cont >= 10) {
                        send_x10_D(cont);
                        cont = cont - 10;
                    } else if (cont >= 5) {
                        send_x5_D(cont);
                        cont = cont - 5;
                    } else {
                        send_D(cont);
                        cont = cont - 1;
                    }

                    clone_to_cpu();

                    //------------------------------------------
                    // decrementamos en 1 el contador despues de clonar el paquete que sera enviado al controlador
                    ContIndexs.write((bit<32>)C_Derecho, cont);

                    // despues de decrementar el contador derecho, validamos si en la proxima ocasion habria mas flujos para enviar,
                    // caso no tenga mas flujos para enviar, se procede a cambiar de carril y actualizar la ventana.
                    if (cont == 0) { 
                        //cambio de carril
                        Carril.write(0,C_Derecho);
                        // se reinicia la ventana de tiempo asignando el actual timestamp 
                        InitTimeWindow.write(0,standard_metadata.ingress_global_timestamp);
                        // se incrementa en 1 el ID de la ventana
                        WindowId.read(meta.WindowNumM, 0);
                        WindowId.write(0,meta.WindowNumM + 1);
                    }
                }


            // ventana activa
            } else {
                // si entra, hay aun estadisticas de la anterior ventana para enviar al controlador.
                if (cont != 0) {
                    //send_D(cont);
                    //cont = cont - 1;
                    if (cont >= 10) {
                        send_x10_D(cont);
                        cont = cont - 10;
                    } else if (cont >= 5) {
                        send_x5_D(cont);
                        cont = cont - 5;
                    } else {
                        send_D(cont);
                        cont = cont - 1;
                    }

                    clone_to_cpu();

                    //------------------------------------------
                    // decrementamos en 1 el contador despues de clonar el paquete que sera enviado al controlador
                    ContIndexs.write((bit<32>)C_Derecho, cont);
                }
            }

        }
        // ###################################################################################################################

        //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Modulo de estadisticas $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        // se muestrea el actual paquete si este es UDP o TCP.
        if (hdr.tcp.isValid() || hdr.udp.isValid()){
            //se calcula el index perteneciente al flujo al que pertenece el paquete
            calculate_hash(); 
            FlowState.read(meta.state,meta.index);
            FlowState.read(meta.state2,meta.index2);

            //--- Es un flujo nuevo y la posicion en cada registrador esta en cero.
            //--- Tanto el subflujo FWD como BWD estan en estado cero.  
            if (meta.state == 0 && meta.state2 == 0) {
                //se instancia el TimeStamp del primer paquete del flujo
                InitTimeFlow.write(meta.index, standard_metadata.ingress_global_timestamp);
                //---- se instancia el valor inicial de todos los registradores----------------------------------------------
                LastTimePacket.write(meta.index, standard_metadata.ingress_global_timestamp);
                // se inicializa con 1 el numero total de paquetes del subflujo
                TotPkts.write(meta.index, 1);

                bit<32> payload;
                if (hdr.tcp.isValid()) {
                    //---- me cuenta el numero de paquetes TCP que pasan por el sw
                    NumPacketsTCP.read(meta.contador,0);                
                    NumPacketsTCP.write(0,meta.contador + 1);

                    // guardamos la longitud en bytes de los datos del segmento TCP (payload)
                    // payload = LenPacketIP - ipv4Header - TCPheader
                    payload = (bit<32>)hdr.ipv4.len - ((bit<32>)hdr.ipv4.ihl * 4) - ((bit<32>)hdr.tcp.data_offset * 4);
                    TotLenPkts.write(meta.index, payload);

                    //la longitud minima y maxima se inicializan con el tamanho del paquete actual como valor de referencia
                    PktLenMin.write(meta.index, payload);
                    PktLenMax.write(meta.index, payload);

                } else {
                    NumPacketsUDP.read(meta.contador,0);                
                    NumPacketsUDP.write(0,meta.contador + 1);

                    // guardamos la longitud en bytes de los datos del datagrama UDP (payload)
                    // payload = LenDatagramaUDP - UDPheader
                    payload = (bit<32>)(hdr.udp.len - 8);
                    TotLenPkts.write(meta.index, payload);

                    //la longitud minima y maxima se inicializan con el tamanho del paquete actual como valor de referencia
                    PktLenMin.write(meta.index, payload);
                    PktLenMax.write(meta.index, payload);
                }

                // se guarda el cuadrado de la longitud del paquete (payload ^ 2)
                TotLenSquare.write(meta.index, (bit<40>)payload * (bit<40>)payload );

                // estadisticas temporales. tiempo de llegada entre paquetes de un flujo indiferente de su dirección.
                TotIAT.write(meta.index, 0);
                TotIATsquare.write(meta.index,0);

                //############################## Guardar los Indexs #################################
                Carril.read(carril,0);
                if (carril == C_Derecho){
                    ContIndexs.read(cont,(bit<32>)C_Derecho);
                    // Guardamos los index del nuevo flujo en el carril derecho
                    indexsFWD0.write(cont,meta.index);
                    indexsBWD0.write(cont,meta.index2);
                    // se incrementa el contador ContIndexs
                    ContIndexs.write((bit<32>)C_Derecho,cont + 1);
                } else {
                    ContIndexs.read(cont,(bit<32>)C_Izquierdo);
                    // Guardamos los index del nuevo flujo en el carril derecho
                    indexsFWD1.write(cont,meta.index);
                    indexsBWD1.write(cont,meta.index2);
                    // se incrementa el contador ContIndexs
                    ContIndexs.write((bit<32>)C_Izquierdo,cont + 1);
                }               
                //###################################################################################

                //se cambia el estado del flujo FWD a 1, osea, flujo activo             
                FlowState.write(meta.index, 1);

            
            //--- El flujo existe y se procede a actualizarse
            //--- (FWD) || (BWD)
            } else if ((meta.state == 1 && meta.state2 == 0) || (meta.state == 0 && meta.state2 == 1)) {

                InitTimeFlow.read(meta.InitTimeFlowM,meta.index);
                LastTimePacket.read(meta.LastTimePacketM, meta.index);
                LastTimePacket.read(meta.LastTimePacketM2, meta.index2);

                // me identifica el subflujo de retorno BWD el cual aun no se ha inicalizado y se procede entonces.
                // se recicla el codigo 
                if (meta.InitTimeFlowM == 0 && meta.state == 0){
                    //se instancia el TimeStamp del primer paquete del flujo
                    InitTimeFlow.write(meta.index, standard_metadata.ingress_global_timestamp);
                    LastTimePacket.write(meta.index, standard_metadata.ingress_global_timestamp);
                    TotPkts.write(meta.index, 1);

                    bit<32> payload;
                    if (hdr.tcp.isValid()) {
                        NumPacketsTCP.read(meta.contador,0);                
                        NumPacketsTCP.write(0,meta.contador + 1);

                        payload = (bit<32>)hdr.ipv4.len - ((bit<32>)hdr.ipv4.ihl * 4) - ((bit<32>)hdr.tcp.data_offset * 4);
                        TotLenPkts.write(meta.index, payload);

                        PktLenMin.write(meta.index, payload);
                        PktLenMax.write(meta.index, payload);

                    } else {
                        NumPacketsUDP.read(meta.contador,0);                
                        NumPacketsUDP.write(0,meta.contador + 1);

                        payload = (bit<32>)(hdr.udp.len - 8);
                        TotLenPkts.write(meta.index, payload);

                        PktLenMin.write(meta.index, payload);
                        PktLenMax.write(meta.index, payload);
                    }

                    TotLenSquare.write(meta.index, (bit<40>)payload * (bit<40>)payload );

                // me actualiza el sublujo FWD o BWD los cuales ya estan previamente inicializados.
                } else {
                    // --- cargar datos en metadatas                    
                    TotPkts.read(meta.TotPktsM, meta.index);
                    TotLenPkts.read(meta.TotLenPktsM, meta.index);
                    PktLenMin.read(meta.PktLenMinM, meta.index);
                    PktLenMax.read(meta.PktLenMaxM, meta.index);
                    TotLenSquare.read(meta.TotLenSquareM, meta.index);

                    // --- se suma un paquete mas al subflujo
                    TotPkts.write(meta.index, meta.TotPktsM + 1);

                    //--------------------------------------------------------------
                    bit<32> payload;
                    // --- tcp
                    if (hdr.tcp.isValid()) {
                        NumPacketsTCP.read(meta.contador,0);                
                        NumPacketsTCP.write(0,meta.contador + 1);

                        // guardamos la longitud en bytes de los datos del segmento TCP (payload)
                        // payload = LenPacketIP - ipv4Header - TCPheader
                        payload = (bit<32>)hdr.ipv4.len - ((bit<32>)hdr.ipv4.ihl * 4) - ((bit<32>)hdr.tcp.data_offset * 4);
                        //Se suma la longitud del payoad al agregado
                        TotLenPkts.write(meta.index, meta.TotLenPktsM + payload);                  

                    // --- udp
                    } else {
                        NumPacketsUDP.read(meta.contador,0);                
                        NumPacketsUDP.write(0,meta.contador + 1);

                        // guardamos la longitud en bytes de los datos del datagrama UDP (payload)
                        // payload = LenDatagramaUDP - UDPheader
                        payload = (bit<32>)(hdr.udp.len - 8);
                        //Se suma la longitud del payoad al agregado
                        TotLenPkts.write(meta.index, meta.TotLenPktsM + payload);
                    }
                    //--------------------------------------------------------------

                    //actualiza el valor de longitud minimo
                    if (payload < meta.PktLenMinM){
                        PktLenMin.write(meta.index, payload);
                    }
                    //actualiza el valor de longitud maximo
                    if (payload > meta.PktLenMaxM){
                        PktLenMax.write(meta.index, payload);
                    }

                    // se suma el cuadrado de la longitud del paquete (payload ^ 2)
                    TotLenSquare.write(meta.index, meta.TotLenSquareM + ((bit<40>)payload * (bit<40>)payload) );

                    //Se actualiza el tiempo de llegada del ultimo paquete con el tiempo de llegada del paquete actual. 
                    LastTimePacket.write(meta.index, standard_metadata.ingress_global_timestamp);
                }

                // ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡ Calculo de Estadisticas indiferente de su direccion ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
                //--- este if me dice si el ultimo paquete del flujo fue en direccion FWD o BWD
                if (meta.LastTimePacketM > meta.LastTimePacketM2) {
                    bit<48> IAT = standard_metadata.ingress_global_timestamp - meta.LastTimePacketM;
                    // --- Me identifica cual es el index que apunta a la direccion FWD. las estadisticas que son
                    // indiferentes a la direccion, serán guardadas en el index que apunta a FWD.
                    if (meta.state == 1) {
                        TotIAT.read(meta.TotIATM, meta.index);
                        TotIAT.write(meta.index, meta.TotIATM + IAT);
                        TotIATsquare.read(meta.TotIATsquareM, meta.index);
                        TotIATsquare.write(meta.index, meta.TotIATsquareM + ((bit<56>)IAT * (bit<56>)IAT) );
                    } else {
                        TotIAT.read(meta.TotIATM, meta.index2);
                        TotIAT.write(meta.index2, meta.TotIATM + IAT);
                        TotIATsquare.read(meta.TotIATsquareM, meta.index2);
                        TotIATsquare.read(meta.TotIATsquareM, meta.index2);
                        TotIATsquare.write(meta.index2, meta.TotIATsquareM + ((bit<56>)IAT * (bit<56>)IAT) );
                    }
                    
                } else {
                    bit<48> IAT = standard_metadata.ingress_global_timestamp - meta.LastTimePacketM2;
                    if (meta.state == 1) {
                        TotIAT.read(meta.TotIATM, meta.index);
                        TotIAT.write(meta.index, meta.TotIATM + IAT);
                        TotIATsquare.read(meta.TotIATsquareM, meta.index);
                        TotIATsquare.write(meta.index, meta.TotIATsquareM + ((bit<56>)IAT * (bit<56>)IAT) );
                    } else {
                        TotIAT.read(meta.TotIATM, meta.index2);
                        TotIAT.write(meta.index2, meta.TotIATM + IAT);
                        TotIATsquare.read(meta.TotIATsquareM, meta.index2);
                        TotIATsquare.write(meta.index2, meta.TotIATsquareM + ((bit<56>)IAT * (bit<56>)IAT) );
                    }

                }
                
                //--- comprobar y actualizar la etiqueta del flujo. 0:Benign, 1:DDoS

                bit<1> T;
                // leemos el array tag en el index FWD.
                if (meta.state == 1){
                    tag.read(T, meta.index);
                    if (T == 0) {
                        if (hdr.ipv4.tag == DDoS_Tag) {
                            tag.write(meta.index, 1);
                        }
                    }
                } else {
                    tag.read(T, meta.index2);
                    if (T == 0) {
                        if (hdr.ipv4.tag == DDoS_Tag) {
                            tag.write(meta.index2, 1);
                        }
                    }
                }
                
                // ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡

            // Entra si state == 1 y state2 == 1, este estado no existe, lo que se traduce en un caso de colision
            } else {
                bit<16> col;
                colitions.read(col,0);
                colitions.write(0,col + 1);
            }

        }
        
        //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    

        if (standard_metadata.ingress_port == CPU_PORT) {
            // Packet received from CPU_PORT, this is a packet-out sent by the
            // controller. Skip table processing, set the egress port as
            // requested by the controller (packet_out header) and remove the
            // packet_out header.
            standard_metadata.egress_spec = hdr.packet_out.egress_port;
            hdr.packet_out.setInvalid();
        } else {
            // Packet received from data plane port.
            // Applies table t_l2_fwd to the packet.            
            if (t_l2_fwd.apply().hit) {
                // Packet hit an entry in t_l2_fwd table. A forwarding action
                // has already been taken. No need to apply other tables, exit
                // this control block.
                return;
            } else {
                //--- Definir puerto de salida de manera estatico -------------------------------------
                standard_metadata.egress_spec = 2;
            }
        }
        
     }
}

//------------------------------------------------------------------------------
// EGRESS PIPELINE
//------------------------------------------------------------------------------

control c_egress(inout headers_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {

    //register <bit<48>> (1) test2;

    apply {

        if (standard_metadata.instance_type == CLONE1) {
            hdr.packet_in.setValid();
            hdr.packet_in.ingress_port = standard_metadata.ingress_port;
            hdr.ethernet.ether_type = CustomEtherType; //se usa un valor de EtherType propio para filtrar

            if (meta.NumFlowsByPacket == 1) {
                hdr.flow.setValid();
                hdr.flow.NumFlowsByPacket = meta.NumFlowsByPacket;
                hdr.flow.F1 = meta.Flow1;

                // se quita el payload del paquete solo indicando el numero de bytes que se quieren transmitir (truncar)
                // packet_in (2 bytes) + ethernet (14 bytes) + flow (44 bytes)  = 60 bytes         
                truncate(60);
            } else if (meta.NumFlowsByPacket == 5) {
                hdr.flow_x5.setValid();
                hdr.flow_x5.NumFlowsByPacket = meta.NumFlowsByPacket;
                hdr.flow_x5.F1 = meta.Flow1;
                hdr.flow_x5.F2 = meta.Flow2;
                hdr.flow_x5.F3 = meta.Flow3;
                hdr.flow_x5.F4 = meta.Flow4;
                hdr.flow_x5.F5 = meta.Flow5;

                // packet_in (2 bytes) + ethernet (14 bytes) + flow_x5 (216 bytes)  = 232 bytes         
                truncate(232);
            } else if (meta.NumFlowsByPacket == 10) {
                hdr.flow_x10.setValid();
                hdr.flow_x10.NumFlowsByPacket = meta.NumFlowsByPacket;
                hdr.flow_x10.F1 = meta.Flow1;
                hdr.flow_x10.F2 = meta.Flow2;
                hdr.flow_x10.F3 = meta.Flow3;
                hdr.flow_x10.F4 = meta.Flow4;
                hdr.flow_x10.F5 = meta.Flow5;
                hdr.flow_x10.F6 = meta.Flow6;
                hdr.flow_x10.F7 = meta.Flow7;
                hdr.flow_x10.F8 = meta.Flow8;
                hdr.flow_x10.F9 = meta.Flow9;
                hdr.flow_x10.F10 = meta.Flow10;

                // packet_in (2 bytes) + ethernet (14 bytes) + flow_x10 (431 bytes)  = 447 bytes         
                truncate(447);    
                
            } else {
                return;
            }
            
            //Se retiran cabeceras que no se necesitan en el controlador.           
            hdr.ipv4.setInvalid();
            hdr.tcp.setInvalid();
            hdr.udp.setInvalid();
        }

    }
}

V1Switch(c_parser(),
         c_verify_checksum(),
         c_ingress(),
         c_egress(),
         c_compute_checksum(),
         c_deparser()) main;
