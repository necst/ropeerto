# Data width converters
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -version 1.1 -module_name dwidth_input_512_64
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {64} CONFIG.M_TDATA_NUM_BYTES {8} CONFIG.TID_WIDTH {6} CONFIG.HAS_TLAST {1} CONFIG.HAS_TKEEP {1} CONFIG.Component_Name {dwidth_input_512_64}] [get_ips dwidth_input_512_64]

create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -version 1.1 -module_name dwidth_output_32_512
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {4} CONFIG.M_TDATA_NUM_BYTES {64} CONFIG.TID_WIDTH {6} CONFIG.HAS_TLAST {1} CONFIG.HAS_TKEEP {1} CONFIG.Component_Name {dwidth_output_32_512}] [get_ips dwidth_output_32_512]