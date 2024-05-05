# Personalized Music Discovery Artificial Neural Network Project
Group Project for CS/ME/ECE 539

## Group Members
- Arushi Renschler Pandey
- Nicolas Ruffolo
- Allen Chien

## Introduction
This project is supposed to build a song recommendation system soley based on a generated user profile and a library of 30,000 Spotify songs. The library of songs is a pool of songs that our various methods of recommendation attempt to pull from to match to the user's profile.

## 30K Spotify Songs
This dataset was found on Kaggle and contained 30,000 songs and associated audio features generated from Kaggle. 

## User Profile Generation
The profiles were generated from a 1.2GB dataset of playlist data from Kaggle. This only included the artist name and the track name. The Spotify API was used to populate this dataset with audio features so it will be able match the 30k dataset so there will be optimal prediction. We distilled down user's playlists by averaging all numerical characteristics, and calculated the mode of all string characters to see which artist appears the most. This results in a profile that we hope accurately reflects the user's preferences.

## Method 1 -> Branch "Method 1"

## Method 2 -> Branch "Method 2"

## Method 3 -> Branch "Method 3"
