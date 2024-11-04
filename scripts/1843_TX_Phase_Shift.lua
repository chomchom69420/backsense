-- BSS_FW = "C:\\ti\\mmwave_dfp_01_02_00_01\\rf_eval\\rf_eval_firmware\\radarss\\xwr18xx_radarss_rprc.bin"
-- MSS_FW = "C:\\ti\\mmwave_dfp_01_02_00_01\\rf_eval\\rf_eval_firmware\\masterss\\xwr18xx_masterss_rprc.bin"

NUM_TX = 3
NUM_RX = 4

START_FREQ = 77 
ADC_START_TIME = 6 
FREQ_SLOPE = 60.012
ADC_SAMPLES = 256
SAMPLE_RATE = 4400
RX_GAIN = 30 

IDLE_TIME = 7
RAMP_END_TIME = 65
CHIRP_LOOPS = 182
PERIODICITY = 150 

autoStart = 1
NumOfProfile = 1
NumOfChirpInLoop = 3
NumOfFrame = 0
profile={}
profile[0]={0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN}
profile[1]={0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN}
profile[2]={1, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN}
profile[3]={2, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN}
ptx = {}
ptx[1]={1,0,0}
ptx[2]={0,0,1}
ptx[3]={0,1,0}
periodicity = 150

-- tx antenna phase shift bits
tx_phase_shift = {0, 0, 0}

info = debug.getinfo(1,'S');
file_path = (info.source);
file_path = string.gsub(file_path, "@","");
i, j = string.find(file_path, "\\.*%.lua")
file_path=string.gsub(file_path, "\\[%w_]-%.lua$","\\");
WriteToLog(file_path.."\n", "red")
-- fw_path   = file_path.."..\\..\\rf_eval_firmware"

-- BSS_FW    = fw_path.."\\radarss\\xwr68xx_radarss.bin"
-- MSS_FW    = fw_path.."\\masterss\\xwr68xx_masterss.bin"

data_path     = file_path.."..\\PostProc"
pkt_log_path  = data_path.."\\pktlogfile.txt"

BSS_FW = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr18xx_radarss.bin"
MSS_FW = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr18xx_masterss.bin"

-- file_path = "C:\\ti\\raw_data_capture"

-- adc_data_path1 = file_path.."\\AOP_adc_data_profile_bf.bin"
-- raw_data_path1 = file_path.."\\AOP_adc_data_profile_bf_Raw_0.bin"
-- pkt_log_path1  = file_path.."\\AOP_profile_bf_pktlogfile.txt"

COM_PORT = 6

--reset and connect
ar1.FullReset()
RSTD.Sleep(2000)
ar1.SOPControl(2)
RSTD.Sleep(1000)
ar1.Connect(COM_PORT,921600,1000)
RSTD.Sleep(2000)
ar1.frequencyBandSelection("77G")
bitopfile = file_path.."\\".."bitoperations.lua"
dofile(bitopfile)

--download BSS FW
if (ar1.DownloadBSSFw(BSS_FW) == 0) then
    WriteToLog("BSS FW Download Success\n", "green")
else
    WriteToLog("BSS FW Download failure\n", "red")
end
RSTD.Sleep(2000)

-- Download MSS FW
if (ar1.DownloadMSSFw(MSS_FW) == 0) then
    WriteToLog("MSS FW Download Success\n", "green")
else
    WriteToLog("MSS FW Download failure\n", "red")
end
RSTD.Sleep(2000)

-- SPI Connect
if (ar1.PowerOn(1, 1000, 0, 0) == 0) then
    WriteToLog("Power On Success\n", "green")
else
   WriteToLog("Power On failure\n", "red")
end
RSTD.Sleep(1000)

-- RF Power UP
if (ar1.RfEnable() == 0) then
    WriteToLog("RF Enable Success\n", "green")
else
    WriteToLog("RF Enable failure\n", "red")
end
RSTD.Sleep(1000)

-- Channel Config
if (ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0) == 0) then
    WriteToLog("ChanNAdcConfig Success\n", "green")
else
    WriteToLog("ChanNAdcConfig failure\n", "red")
end
RSTD.Sleep(1000)

-- ADC Config
if (ar1.LPModConfig(0, 0) == 0) then
	WriteToLog("Regualar mode Cfg Success\n", "green")
else
	WriteToLog("Regualar mode Cfg failure\n", "red")
end
RSTD.Sleep(1000)

-- LDO Bypass Enable
if (ar1.RfLdoBypassConfig(0x3) == 0) then
    WriteToLog("RfLdoBypass Cfg Success\n", "green")
else
    WriteToLog("RfLdoBypass Cfg failure\n", "red")
end
RSTD.Sleep(2000)

-- Enable Per Chirp Phase Shifter
if (ar1.SetMiscConfig(1) == 0) then
	WriteToLog("Per Chirp Phase Shifter Enable Success\n", "green")
else
	WriteToLog("Per Chirp Phase Shifter Enable failure\n", "red")
end
RSTD.Sleep(1000)

-- RF Init
if (ar1.RfInit() == 0) then
    WriteToLog("RfInit Success\n", "green")
else
    WriteToLog("RfInit failure\n", "red")
end
RSTD.Sleep(1000)

-- Data Path Config
-- if (ar1.DataPathConfig(513, 1216644097, 0) == 0) then
if (ar1.DataPathConfig(1, 1, 0) == 0) then
    WriteToLog("DataPathConfig Success\n", "green")
else
    WriteToLog("DataPathConfig failure\n", "red")
end
RSTD.Sleep(1000)

if (ar1.LvdsClkConfig(1, 1) == 0) then
    WriteToLog("LvdsClkConfig Success\n", "green")
else
    WriteToLog("LvdsClkConfig failure\n", "red")
end
RSTD.Sleep(1000)

if (ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0) == 0) then
    WriteToLog("LVDSLaneConfig Success\n", "green")
else
    WriteToLog("LVDSLaneConfig failure\n", "red")
end
RSTD.Sleep(1000)

-- Profile Config
for n = 0,NumOfProfile-1 do
    if(ar1.ProfileConfig(profile[n][1],profile[n][2],profile[n][3],profile[n][4],profile[n][5],profile[n][6],profile[n][7],profile[n][8],profile[n][9],profile[n][10],profile[n][11],profile[n][12],profile[n][13],profile[n][14],profile[n][15],profile[n][16],profile[n][17],profile[n][18]) == 0) then
        WriteToLog("ProfileConfig No."..n.."Success\n", "green")
    else
        WriteToLog("ProfileConfig No."..n.."failure\n", "red")
    end      
end
RSTD.Sleep(1000)

-- Chirp config 
for i = 0, NumOfChirpInLoop - 1 do
    p = 0
    tx1 = ptx[i+1][1]
    tx2 = ptx[i+1][2]
    tx3 = ptx[i+1][3]
    if (ar1.ChirpConfig(i, i, p, 0, 0, 0, 0, tx1, tx2, tx3) == 0) then
        WriteToLog("ChirpConfig No."..i.." Success\n", "green")
    else
        WriteToLog("ChirpConfig No."..i.." failure\n", "red")
    end
end
RSTD.Sleep(1000)

-- Configure Chirp Based Phase Shifter
if (ar1.SetPerChirpPhaseShifterConfig(0, 0, tx_phase_shift[0], tx_phase_shift[1], tx_phase_shift[2]) == 0) then
    WriteToLog("Chirp Based Phase Shifter Configuration Success\n", "green")
else
    WriteToLog("Chirp Based Phase Shifter Configuration failure\n", "red")
end
RSTD.Sleep(1000)

-- Frame Config 
if (ar1.FrameConfig(0, NumOfChirpInLoop-1, NumOfFrame, CHIRP_LOOPS,periodicity, 0, 0, 1) == 0) then
    WriteToLog("FrameConfig Success\n", "green")
else
    WriteToLog("FrameConfig failure\n", "red")
end
RSTD.Sleep(1000)

-- Connect DCA1000
if (ar1.SelectCaptureDevice("DCA1000") == 0) then
    WriteToLog("SelectCaptureDevice Success\n", "green")
else
    WriteToLog("SelectCaptureDevice failure\n", "red")
end
RSTD.Sleep(1000)

if (ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098) == 0) then
    WriteToLog("CaptureCardConfig_EthInit Success\n", "green")
else
    WriteToLog("CaptureCardConfig_EthInit failure\n", "red")
end
RSTD.Sleep(1000)

if (ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30) == 0) then
    WriteToLog("CaptureCardConfig_Mode Success\n", "green")
else
    WriteToLog("CaptureCardConfig_Mode failure\n", "red")
end
RSTD.Sleep(1000)

if (ar1.CaptureCardConfig_PacketDelay(25) == 0) then
    WriteToLog("CaptureCardConfig_PacketDelay Success\n", "green")
else
    WriteToLog("CaptureCardConfig_PacketDelay failure\n", "red")
end
RSTD.Sleep(1000)

----------------------------- Capture 1 -----------------------------
-- --Start Record ADC data
-- ar1.CaptureCardConfig_StartRecord(adc_data_path1, 1)
-- RSTD.Sleep(1000)

-- --Trigger frame
-- ar1.StartFrame()
-- RSTD.Sleep(15000)

-- --Packet reorder utility processing the Raw_ADC_data
-- WriteToLog("Please wait for a few seconds for Packet reorder utility processing .....!!!! \n", "green")
-- ar1.PacketReorderZeroFill(raw_data_path1, adc_data_path1, pkt_log_path1)
-- RSTD.Sleep(10000)
-- WriteToLog("Packet reorder utility processing done.....!!!! \n", "green")

tstring = os.date('%Y-%m-%d_%H;%M;%S', t0)
local socket1 = require"socket"
local t0 = socket1.gettime()
os.execute("mkdir \"" ..data_path.."\\".. tstring.."\"")
Raw_data_path = data_path.."\\"..tstring.."\\adc_data_Raw_0.bin"
adc_data_path = data_path.."\\"..tstring.."\\adc_data.bin"
ar1.CaptureCardConfig_StartRecord(adc_data_path, 0)
RSTD.Sleep(1000)

if (autoStart > 0 ) then
    
    local socket1 = require"socket"
    local t0 = socket1.gettime()
    print(t0)
    f = assert(io.open(data_path.."\\".. tstring.."\\".."timestamp.txt",'w'))
    f:write(t0)
    f:close()

    ar1.StartFrame()
    RSTD.Sleep(2000)
end
