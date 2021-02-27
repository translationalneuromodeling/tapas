function [byte] = inp(address)

global cogent;

byte = io64(cogent.io.ioObj,address);
