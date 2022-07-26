//// How to use
//// Open macro from ImageJ menu
//// Change parameter setting here, and run entire macro
//// For input folder browser, choose the input image folder 
//// It takes ~20 seconds to run this macro

print("\\Clear");


// Parameters setting
source_image="merFISH_02_007_01_wavelength_561.TIFF";  // change parameter here
target_image="merFISH_02_007_01_wavelength_647.TIFF";  // change parameter here
suffixe=".TIFF";






print("----------------------------------------------------------------");
//dir = getArgument;
dir=getDirectory("mouse"); // open a browser, allow you to choose input directory
//dir = "yourlocaldir/bleed_throught_validate/raw/";  
// ex: dir="/Users/htran/Documents/storage_tmp/merfish_XP2059/bleed_throught_validate/raw/";
if (dir=="") 
	exit ("No argument!");

print("Working dir: "+dir+"\n");

results_dir=File.getParent(dir)+"/results/";
if(!File.exists(results_dir)) 
      File.mkdir(results_dir);

            


					
					
short_name_source = substring(source_image,0,lastIndexOf(source_image,suffixe));
short_name_target = substring(target_image,0,lastIndexOf(target_image,suffixe));
IJ.log(short_name_source);
IJ.log(short_name_target);

// Source image first
print("Loading image: "+source_image);
open(dir+source_image);																				
selectWindow(source_image);
run("Enhance Contrast", "saturated=0.35");
if (bitDepth > 8) {run("8-bit");}
//run("Median...", "radius=2");  // in case you see lots of noises detected as signals
run("Convolve...", "text1=[-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n] normalize");


////only signals with intensity values from 250 to 255 are considered as signals, from my observation of spots and noise in images. 
////You can use other thresholds, this macro just give an estimation, not provide accurate results for publication. 
setThreshold(250, 255, "raw"); 
//setThreshold(250, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Grays");

selectWindow(source_image);
bin_source=short_name_source+"_BINARY";  //can open zip file from ImageJ to have tif format
saveAs("ZIP", results_dir+bin_source+".zip");
print("Deconvolution done!");



					
print("Loading image: "+target_image);
open(dir+target_image);
selectWindow(target_image);
run("Enhance Contrast", "saturated=0.35");
if (bitDepth > 8) {run("8-bit");}

// ATTENTION: in case you see lots of noises detected as signals, because this image contains large amount of noises --> need to use median filter here
// If you don't see lots of noise, please comment median filter.  
run("Median...", "radius=2"); 

run("Convolve...", "text1=[-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n] normalize");
setThreshold(250, 255, "raw");
//setThreshold(250, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Grays");

selectWindow(target_image);
bin_target=short_name_target+"_BINARY"; //can open zip file from ImageJ to have tif format
saveAs("ZIP",results_dir+bin_target+".zip");
print("Deconvolution done!");






// histogram here

//// First, extracting signals that are overlapped in source and target images
imageCalculator("AND create", bin_source+".tif", bin_target+".tif");
selectWindow("Result of "+bin_source+".tif");
output_image="bleedthrought_signals"; //can open zip file from ImageJ to have tif format
saveAs("ZIP",results_dir+output_image+".zip");


//Counting number of spots in source image
selectWindow(bin_source+".tif");
nBins = 256; 
getHistogram(values, counts, nBins);
source_spots=counts[255]; // number of spots
IJ.log("Source image is: "+source_image);
IJ.log("Number of spots in source image is: "+source_spots);


//Counting number of spots in bleedthrought image
selectWindow(output_image+".tif");
getHistogram(values, counts, nBins);
bleedthrought_spots=counts[255]; // number of spots
IJ.log("Number of spots that bleed throught other wavelength channel is: "+bleedthrought_spots);


//Counting number of spots in target image
selectWindow(bin_target+".tif");
getHistogram(values, counts, nBins);
target_spots=counts[255]; // number of spots
IJ.log("Target image is: "+target_image);
IJ.log("Number of spots in target image is: "+target_spots);


pct_bleed=100*bleedthrought_spots/source_spots;
IJ.log("Percentage of bleed throught is: "+pct_bleed);

pct_bleed_target=100*bleedthrought_spots/target_spots;
IJ.log("Percentage of bleed throught is: "+pct_bleed_target);


// Save results into a csv file
setResult("source_img", 0, source_image);
setResult("target_img", 0, target_image);
setResult("pct_bleedthrought_source", 0, pct_bleed);
setResult("pct_bleedthrought_target", 0, pct_bleed_target);
updateResults();

selectWindow("Results");
saveAs("Results",results_dir+"bleed_throught_report.csv");
selectWindow("Results");
run("Close");
print("Save output into the folder: "+results_dir);

run("Close All");

print("Completed");
print("----------------------------------------------------------------");


