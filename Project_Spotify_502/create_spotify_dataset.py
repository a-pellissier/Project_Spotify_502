from Project_Spotify_502 import utils_spotify as u

if __name__ == "__main__":
    print ('entering run')
    dl_data = u.DataSpotify()

    dl_data.generate_spot_data()
