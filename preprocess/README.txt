HomePage: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
Download Link (Faces only): https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar

Usage
For both the IMDb and Wikipedia images we provide a separate .mat file which can be loaded with Matlab containing all the meta information. The format is as follows:

- dob: date of birth (Matlab serial date number)

- photo_taken: year when the photo was taken

- full_path: path to file

- gender: 0 for female and 1 for male, NaN if unknown

- name: name of the celebrity

- face_location: location of the face. To crop the face in Matlab run
  img(face_location(2):face_location(4),face_location(1):face_location(3),:)

- face_score: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image

- second_face_score: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.

- celeb_names (IMDB only): list of all celebrity names

- celeb_id (IMDB only): index of celebrity name

The age of a person can be calculated based on the date of birth and the time when the photo was taken (note that we assume that the photo was taken in the middle of the year):
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob); 