localparam integer W=16,H=8,FRAMES=3,INIT_WORDS=3800,RECORDS=1696,MEMORY_BYTES=103808;
localparam integer D0=2,D1=3,D2=4,D3=4,D4=3,D5=2,D6=2;
localparam [63:0] CONFIG_BASE=64'h16000,CACHE_BASE=64'h15000;
function automatic [63:0] descriptor(input integer idx); begin case(idx)
0: descriptor=64'd8192;
1: descriptor=64'd8256;
2: descriptor=64'd8320;
default: descriptor=0; endcase end endfunction
function automatic [63:0] output_base(input integer idx); begin case(idx)
0: output_base=64'd77824;
1: output_base=64'd81920;
default: output_base=0; endcase end endfunction
function automatic [63:0] gold_writes(input integer idx); begin case(idx)
0: gold_writes=64'd2;
1: gold_writes=64'd4;
2: gold_writes=64'd4;
default: gold_writes=0; endcase end endfunction
function automatic [63:0] gold_skips(input integer idx); begin case(idx)
0: gold_skips=64'd0;
1: gold_skips=64'd0;
2: gold_skips=64'd2;
default: gold_skips=0; endcase end endfunction
