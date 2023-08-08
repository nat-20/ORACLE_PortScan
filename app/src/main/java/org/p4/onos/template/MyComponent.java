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

package org.p4.onos.template;


import org.apache.commons.collections.map.HashedMap;
import org.onlab.packet.Ethernet;
import org.onosproject.core.ApplicationId;
import org.onosproject.mastership.MastershipService;
import org.onosproject.net.DeviceId;
import org.onosproject.net.PortNumber;
import org.onosproject.net.config.NetworkConfigService;
import org.onosproject.net.device.*;
import org.onosproject.net.flow.*;
import org.onosproject.net.group.GroupDescription;
import org.onosproject.net.group.GroupService;
import org.onosproject.net.intf.InterfaceService;
import org.onosproject.net.packet.*;
import org.onosproject.net.pi.model.*;
import org.onosproject.net.pi.runtime.*;
import org.onosproject.net.pi.service.PiPipeconfService;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.p4.onos.template.common.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.onosproject.p4runtime.api.P4RuntimeController;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

import static org.p4.onos.template.AppConstants.CPU_CLONE_SESSION_ID;


/**
 * App component that configures devices to provide L2 bridging capabilities.
 */
@Component(
        immediate = true,
        enabled = true,
        service = MyComponent.class
)
public class MyComponent {

    private final Logger log = LoggerFactory.getLogger(getClass());


    private ApplicationId appId;

    //--------------------------------------------------------------------------
    // ONOS CORE SERVICE BINDING
    //
    // These variables are set by the Karaf runtime environment before calling
    // the activate() method.
    //--------------------------------------------------------------------------

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private DeviceService deviceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private InterfaceService interfaceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private NetworkConfigService configService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private FlowRuleService flowRuleService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private MastershipService mastershipService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private MainComponent mainComponent;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private PiPipeconfService piPipeconfService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private P4RuntimeController p4RuntimeController;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected PacketService packetService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private GroupService groupService;


    private final DeviceListener deviceListener = new MyComponent.InternalDeviceListener();
    private MonitoringPacketProcessor processor = new MonitoringPacketProcessor();
    Map<String, DeviceId> swIDs = new HashMap<String, DeviceId>();
    GroupDescription cloneGroup;

    private final short CustomEtherType = 26985; //En HEX: 0x6969
    private static final PiRegisterId Rep_ID = PiRegisterId.of("c_ingress.Repitis");
    private static final PiRegisterCellId Rep_Cell_ID = PiRegisterCellId.of(Rep_ID,1);
    private int contador = 0;
    private CrearCSV file;
    private CrearCSV2 file2;
    private final String coma = ",";

    URL url;
    HttpURLConnection con;

    //--------------------------------------------------------------------------
    // COMPONENT ACTIVATION.
    //
    // When loading/unloading the app the Karaf runtime environment will call
    // activate()/deactivate().
    //--------------------------------------------------------------------------

    @Activate
    protected void activate() {
        appId = mainComponent.getAppId();
        deviceService.addListener(deviceListener);
        log.info("Started");
        packetService.addProcessor(processor, PacketProcessor.director(1));

        //-----------------Crear el archivo de excel CSV-------------------------
        file = new CrearCSV();
        file2 = new CrearCSV2();
        String nombreDeArchivo = "/home/sdn/archivo.csv";
        String Statistics = "/home/sdn/statistics.csv";


        file.crearArchivoCSV(nombreDeArchivo);
        file2.crearArchivoCSV(Statistics);
        log.info("/////////////////////////////////////////");
    }

    @Deactivate
    protected void deactivate() {
        deviceService.removeListener(deviceListener);
        packetService.removeProcessor(processor);
        //groupService.removeGroup(swIDs.get("s1"),cloneGroup.appCookie(),appId);
        log.info("Stopped");
        file.CerrarArchivoCSV();
        file2.CerrarArchivoCSV();
    }


    public void p4statisticsSW1(){
        //log.info("Contador de estadisticas recibidas = {}",contador);

        double f11 = 3.0;
        double f12 = 9.0;
        double f13 = 0.0;
        double f14 = 0.0;

        double f21 = 9.0;
        double f22 = 18.0;
        double f23 = 0.0;
        double f24 = 0.0;

        double f31 = 4.0;
        double f32 = 18.0;
        double f33 = 6.0;
        double f34 = 6.0;

        String jsonInputString = "["+f11+","+f12+","+f13+","+f14+",1]";
        jsonInputString = jsonInputString + ",["+f21+","+f22+","+f23+","+f24+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+f31+","+f32+","+f33+","+f34+",0]";
        jsonInputString = jsonInputString + ",["+System.currentTimeMillis()+","+333+","+0+","+0+","+0+"]";
        jsonInputString = "["+jsonInputString+"]";

        log.info(jsonInputString);
        SendToClassifier(jsonInputString);
    }

    public void SendToClassifier(String jsonInputString){
        try {
            String u = "http://0.0.0.0:5000/clasificar";
            url = new URL(u);
            con = (HttpURLConnection)url.openConnection();
            con.setRequestProperty("Content-Type", "application/json; utf-8");
            con.setRequestProperty("Accept", "application/json");
            con.setDoOutput(true);
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try(OutputStream os = con.getOutputStream()) {
            byte[] input = jsonInputString.getBytes("utf-8");
            os.write(input, 0, input.length);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try(BufferedReader br = new BufferedReader(
                new InputStreamReader(con.getInputStream(), "utf-8"))) {
            StringBuilder response = new StringBuilder();
            String responseLine = null;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            System.out.println(response.toString());
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        con.disconnect();
    }


    public class InternalDeviceListener implements DeviceListener {

        @Override
        public boolean isRelevant(DeviceEvent event) {
            switch (event.type()) {
                case DEVICE_ADDED:
                case DEVICE_AVAILABILITY_CHANGED:
                    break;
                default:
                    // Ignore other events.
                    return false;
            }
            // Process only if this controller instance is the master.
            final DeviceId deviceId = event.subject().id();
            return mastershipService.isLocalMaster(deviceId);
        }

        @Override
        public void event(DeviceEvent event) {
            final DeviceId deviceId = event.subject().id();
            if (deviceService.isAvailable(deviceId)) {
                // A P4Runtime device is considered available in ONOS when there
                // is a StreamChannel session open and the pipeline
                // configuration has been set.
                mainComponent.getExecutorService().execute(() -> {
                    log.info("{} event! deviceId={}", event.type(), deviceId);
                    storeIdSwitches(deviceId);
                });
            }
        }

    }

    public void storeIdSwitches(DeviceId deviceId){
        String id = deviceId.toString();
        String key = id.substring(id.length()-2, id.length());
        swIDs.put(key,deviceId);
        log.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
        log.info("Device detectado {}= ",deviceId);

        //---se llama esta funcion para instanciar un grupo "clone" que seria lo analogo a un ID_SESSION
        //---definido con el comando "mirring_add", pero no esta funcionando bien. (PRE)
        CloneSessionID(deviceId);
    }

    public void CloneSessionID(DeviceId deviceId){
        cloneGroup = Utils.buildCloneGroup(
                appId,
                deviceId,
                CPU_CLONE_SESSION_ID,
                // Ports where to clone the packet.
                // Just controller in this case.
                Collections.singleton(PortNumber.CONTROLLER));
        log.info("############ y aqui vamos ###############");
        groupService.addGroup(cloneGroup);
    }

    private class MonitoringPacketProcessor implements PacketProcessor {
        @Override
        public void process(PacketContext context){
            InboundPacket pkt = context.inPacket();
            //ConnectPoint SwSource = pkt.receivedFrom();
            Ethernet ethPkt = pkt.parsed();

            if (ethPkt == null) {
                return;
            }

            if (ethPkt.getEtherType() == CustomEtherType) {
                long arrived = System.currentTimeMillis();//Tiempo en que se recibe el paquete en el controlador

                //Se obtiene lo encapsulado por la cabecera Ethernet y me lo muestra en forma de array
                //donde cada posicion (numero) me representa 1 Byte.
                byte[] payload = ethPkt.getPayload().serialize();

                int SizeCustomHeader = 43;
                int NumFlowsByPktsH;
                byte[] FlowDurationH = new byte[6];
                byte[] TotPktsH_FWD = new byte[4];
                byte[] TotPktsH_BWD = new byte[4];
                byte[] TotLenPktsH_FWD = new byte[4];
                byte[] TotLenPktsH_BWD = new byte[4];
                byte[] TotLenSquareH_BWD = new byte[5];
                byte[] TotIATH = new byte[6];
                byte[] TotIATsquareH = new byte[7];
                byte[] WindowH = new byte[2];
                byte tagH = 0;


                NumFlowsByPktsH = payload[0] & 0xff; //primer byte de la custom header

                //log.info("num flujos by packet: {}",NumFlowsByPktsH);

                //log.info("////////////////////////////////////");
                //log.info("el paquete completo: {}", ethPkt.serialize());
                //log.info("event! Payload del paquete : {}", payload);
                //log.info("event! tamanho del payload : {}", payload.length);

                contador = contador + 1;
                String contenedor = "";

                for (int p = 0; p < NumFlowsByPktsH; p++){

                    for (int i = 0; i < SizeCustomHeader; i++){
                        //log.info("{}",i);
                        if (i < 6){
                            FlowDurationH[i] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 6 && i < 10){
                            TotPktsH_FWD[i-6] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 10 && i < 14){
                            TotPktsH_BWD[i-10] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 14 && i < 18){
                            TotLenPktsH_FWD[i-14] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 18 && i < 22){
                            TotLenPktsH_BWD[i-18] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 22 && i < 27){
                            TotLenSquareH_BWD[i-22] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 27 && i < 33){
                            TotIATH[i-27] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 33 && i < 40){
                            TotIATsquareH[i-33] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 40 && i < 42){
                            WindowH[i-40] = payload[i+1+(p*SizeCustomHeader)];
                        }
                        if (i >= 42 && i < 43){
                            tagH = payload[i+1+(p*SizeCustomHeader)];
                        }
                    }
                    //log.info("event! TotIATsquareH : {}", TotIATsquareH);

                    double FlowDuration =  Byte2Long(FlowDurationH);
                    //log.info("Flow Duration: {}", FlowDuration);
                    double TotPkts_FWD =  Byte2Long(TotPktsH_FWD);
                    //log.info("TotPkts_FWD: {}", TotPkts_FWD);
                    double TotPkts_BWD =  Byte2Long(TotPktsH_BWD);
                    //log.info("TotPkts_BWD: {}", TotPkts_BWD);
                    double TotLenPkts_FWD =  Byte2Long(TotLenPktsH_FWD);
                    //log.info("TotLenPkts_FWD: {}",TotLenPkts_FWD);
                    double TotLenPkts_BWD =  Byte2Long(TotLenPktsH_BWD);
                    //log.info("TotLenPkts_BWD: {}",TotLenPkts_BWD);
                    double TotLenSquare_BWD =  Byte2Long(TotLenSquareH_BWD);
                    //log.info("TotLenSquare: {}", TotLenSquare_BWD);
                    double TotIAT =  Byte2Long(TotIATH);
                    //log.info("TotIAT: {}", TotIAT);
                    double TotIATsquare =  Byte2Long(TotIATsquareH);
                    //log.info("TotIATsquare: {}", TotIATsquare);
                    int Window = byte2int_16(WindowH);
                    //log.info("Window: {}", Window);
                    int tag = tagH & 0xff;
                    //log.info("Flow: {}",tag);
                    //log.info("-------------------------------");

                    String features = GetFeatures(contador, FlowDuration, TotPkts_FWD, TotPkts_BWD, TotLenPkts_FWD, TotLenPkts_BWD,
                            TotLenSquare_BWD, TotIAT, TotIATsquare, arrived, Window, tag);

                    if (p == 0) {
                        contenedor = features;
                    } else {
                        contenedor = contenedor + coma + features;
                    }

                    /*file.EscribirCSV(Integer.toString(contador),Double.toString(FlowDuration),Double.toString(TotPkts_FWD),
                            Double.toString(TotPkts_BWD),Double.toString(TotLenPkts_FWD),Double.toString(TotLenPkts_BWD),
                            Double.toString(TotLenSquare_BWD),Double.toString(TotIAT),Double.toString(TotIATsquare),
                            Integer.toString(Window),Long.toString(arrived), Integer.toString(tag));*/

                }
                long TimeSpentONOS = System.currentTimeMillis() - arrived;
                contenedor = contenedor + coma + "["+arrived+","+TimeSpentONOS+","+NumFlowsByPktsH+","+0+","+0+"]";
                contenedor = "["+contenedor+"]";
                SendToClassifier(contenedor); //enviar conjunto de flujos a clasificar
            }
        }
    }

    //--------------- Metodos para convertir de Bytes a Decimal -------------------------
    public double Byte2Long (byte[] NumByte){
        long value = 0;
        for (int i = 0; i < NumByte.length; i++) {
            value = (value << 8) + (NumByte[i] & 0xff);
        }
        return (double)value;
    }

    public int byte2int_16 (byte[] NumByte) {
        int numInt = (NumByte[0] << 8 ) & 0xff00 |
                     (NumByte[1])       & 0x00ff;
        return numInt;
    }


    //--------------------- Calculo de Features ---------------------------------------
    public String GetFeatures (int contador, double FlowDuration, double TotPkts_FWD, double TotPkts_BWD, double TotLenPkts_FWD,
                             double TotLenPkts_BWD, double TotLenSquare_BWD, double TotIAT, double TotIATsquare,
                             long time, int Window, int tag) {

        double AvgPackSize = 0;
        if ((TotPkts_FWD + TotPkts_BWD - 1) > 0) {
            AvgPackSize = (TotLenPkts_FWD + TotLenPkts_BWD) / (TotPkts_FWD + TotPkts_BWD - 1);
        } else {
            AvgPackSize = (TotLenPkts_FWD + TotLenPkts_BWD) / (TotPkts_FWD + TotPkts_BWD);
        }

        //-----
        double PackLenSTD_BWD = 0;
        if (TotPkts_BWD - 1 > 0){
            double PackLenMean_BWD = TotLenPkts_BWD / (TotPkts_BWD);
            //log.info("PackLenMean_BWD: {}", PackLenMean_BWD);
            PackLenSTD_BWD = Math.sqrt( (TotLenSquare_BWD - (2*PackLenMean_BWD*TotLenPkts_BWD) + ((TotPkts_BWD)*PackLenMean_BWD*PackLenMean_BWD))/(TotPkts_BWD - 1) );
            //log.info("PackLenSTD_BWD: ",PackLenSTD_BWD);
        } else if (TotPkts_BWD - 1 == 0){
            double PackLenMean_BWD = TotLenPkts_BWD / (TotPkts_BWD);
            PackLenSTD_BWD = Math.sqrt( (TotLenSquare_BWD - (2*PackLenMean_BWD*TotLenPkts_BWD) + ((TotPkts_BWD)*PackLenMean_BWD*PackLenMean_BWD))/(TotPkts_BWD) );
        }
        //-----

        double nIATs = TotPkts_FWD + TotPkts_BWD - 1;
        double FlowIATstd = 0;
        if (nIATs >= 2) {
            double TotIATmean = TotIAT/nIATs;
            //log.info("TotIATmean: {}",TotIATmean);
            double T2 = (2*TotIATmean*TotIAT);
            double T3 = (nIATs*TotIATmean*TotIATmean);
            FlowIATstd = Math.sqrt( ( TotIATsquare - T2 + T3 )/(nIATs-1) );
        } else if (nIATs > 1){
            double TotIATmean = TotIAT/nIATs;
            double T2 = (2*TotIATmean*TotIAT);
            double T3 = (nIATs*TotIATmean*TotIATmean);
            FlowIATstd = Math.sqrt( ( TotIATsquare - T2 + T3 )/(nIATs-1) );
        }
        //log.info("FlowIATstd: {}", FlowIATstd);

        /*file2.EscribirCSV(Integer.toString(contador), Double.toString(FlowDuration), Double.toString(AvgPackSize),
                Double.toString(PackLenSTD_BWD), Double.toString(FlowIATstd), Long.toString(Window), Long.toString(time),
                Integer.toString(tag));
*/
        //return "["+FlowDuration+","+AvgPackSize+","+PackLenSTD_BWD+","+FlowIATstd+","+tag+"]";
	    return "["+PackLenSTD_BWD+","+AvgPackSize+","+FlowDuration+","+FlowIATstd+","+tag+"]";
    }


    //------- Clase para almacenar las estadisticas por flujo en un archivo CSV --------
    public class CrearCSV {

        private FileWriter fw;
        private final String delim = ",";
        private final String NEXT_LINE = "\n";

        private void crearArchivoCSV(String file) {
            try {
                fw = new FileWriter(file);
                fw.append("#").append(delim);
                fw.append("FlowDuration[us]").append(delim);
                fw.append("TotPkts_FWD").append(delim);
                fw.append("TotPkts_BWD").append(delim);
                fw.append("TotLenPkts_FWD").append(delim);
                fw.append("TotLenPkts_BWD").append(delim);
                fw.append("TotLenSquare").append(delim);
                fw.append("TotIAT").append(delim);
                fw.append("TotIATsquare").append(delim);
                fw.append("time").append(delim);
                fw.append("WindowID").append(delim);
                fw.append("Etiqueta").append(NEXT_LINE);
                fw.flush();
            } catch (IOException e) {
                log.info(e.getMessage());
            }
        }

        private void EscribirCSV(String contador, String FlowDuration, String TotPkts_FWD, String TotPkts_BWD,
                                 String TotLenPkts_FWD, String TotLenPkts_BWD, String TotLenSquare, String TotIAT,
                                 String TotIATsquare, String Window, String time, String tag){

            try {
                fw.append(contador).append(delim);
                fw.append(FlowDuration).append(delim);
                fw.append(TotPkts_FWD).append(delim);
                fw.append(TotPkts_BWD).append(delim);
                fw.append(TotLenPkts_FWD).append(delim);
                fw.append(TotLenPkts_BWD).append(delim);
                fw.append(TotLenSquare).append(delim);
                fw.append(TotIAT).append(delim);
                fw.append(TotIATsquare).append(delim);
                fw.append(time).append(delim);
                fw.append(Window).append(delim);
                fw.append(tag).append(NEXT_LINE);
                fw.flush();
            } catch (IOException e) {
                log.info(e.getMessage());
            }
        }

        private void CerrarArchivoCSV(){
            try {
                fw.close();
            }catch (IOException e){
                log.info(e.getMessage());
            }
        }
    }


    //---- Clase para almacenar las FEATURES calculadas por flujo en un archivo CSV ------
    public class CrearCSV2 {

        private FileWriter fw;
        private final String delim = ",";
        private final String NEXT_LINE = "\n";

        private void crearArchivoCSV(String file) {
            try {
                fw = new FileWriter(file);
                fw.append("#").append(delim);
                fw.append("FlowDuration[us]").append(delim);
                fw.append("AvgPackSize").append(delim);
                fw.append("PackLenSTD_BWD").append(delim);
                fw.append("FlowIATstd").append(delim);
                fw.append("time").append(delim);
                fw.append("WindowID").append(delim);
                fw.append("Etiqueta").append(NEXT_LINE);
                fw.flush();
            } catch (IOException e) {
                log.info(e.getMessage());
            }
        }

        private void EscribirCSV(String contador, String FlowDuration, String AvgPackSize, String PackLenSTD_BWD,
                                 String FlowIATstd, String Window, String time, String tag){

            try {
                fw.append(contador).append(delim);
                fw.append(FlowDuration).append(delim);
                fw.append(AvgPackSize).append(delim);
                fw.append(PackLenSTD_BWD).append(delim);
                fw.append(FlowIATstd).append(delim);
                fw.append(time).append(delim);
                fw.append(Window).append(delim);
                fw.append(tag).append(NEXT_LINE);
                fw.flush();
            } catch (IOException e) {
                log.info(e.getMessage());
            }
        }

        private void CerrarArchivoCSV(){
            try {
                fw.close();
            }catch (IOException e){
                log.info(e.getMessage());
            }
        }
    }

}
