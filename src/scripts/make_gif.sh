 #!/usr/bin/env bash
# run from src
experiment_name=$1
images_number=$2
echo $experiment_name
echo $images_number
bgans_path="../out/images/$experiment_name"
echo $bgans_path
for i in `seq -w 00 $(($images_number-1))`;
do
  echo "Processing sample $i..."
  convert -delay 8 -loop 0 $bgans_path/*_$i.png $bgans_path/animation_$i.gif
done   
