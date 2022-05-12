#!/usr/bin/env bash
while getopts a:n:u:d: flag
do
    case "${flag}" in
        a) author=${OPTARG};;
        n) name=${OPTARG};;
        u) urlname=${OPTARG};;
        d) description=${OPTARG};;
    esac
done

echo "Author: $author";
echo "Project Name: $name";
echo "Project URL name: $urlname";
echo "Description: $description";

echo "Renaming project..."

original_author="pritchardn"
original_name="dlg_lowpass_components"
original_urlname="dlg_lowpass_components"
original_description="Awesome dlg_lowpass_components created by pritchardn"
# for filename in $(find . -name "*.*") 
for filename in $(git ls-files) 
do
    sed -i "s/$original_author/$author/g" $filename
    sed -i "s/$original_name/$name/g" $filename
    sed -i "s/$original_urlname/$urlname/g" $filename
    sed -i "s/$original_description/$description/g" $filename
    echo "Renamed $filename"
done

mv dlg_lowpass_components $name

# remove the template related documents
rm ABOUT_THIS_TEMPLATE.md
rm CONTRIBUTING.md
rm HISTORY.md && touch HISTORY.md
# remove template stuff from README
sed -ibak '1,/^<!--  DELETE THE LINES ABOVE THIS AND WRITE YOUR PROJECT README BELOW -->$/d' README.md
# This command runs only once on GHA!
rm -rf .github/template.yml
