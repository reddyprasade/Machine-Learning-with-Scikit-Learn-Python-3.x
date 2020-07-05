import math as m


print("Independent Events Probabilitys")
Play_yes = 9/14
Play_no = 5/14
Entropy_Play = -Play_no*m.log(Play_no,2)  -  Play_yes*m.log(Play_yes,2)
print("Entropy of Play",Entropy_Play)


Windy_true = 6/14
Windy_false = 8/14
Entropy_Windy = -Windy_false*m.log(Windy_false,2)  -  Windy_true*m.log(Windy_true,2)
print("Entropy of Windy",Entropy_Windy)

Humidity_High = 7/14
Humidity_Normal = 7/14
Entropy_Humidity = -Humidity_Normal*m.log(Humidity_Normal,2)  -  Humidity_High*m.log(Humidity_High,2)
print("Entropy of Humidity",Entropy_Humidity)

print('*'*30)


print("Entropy of Outlook Feature and Play Glof ")

sunny_yes =3/5
sunny_no = 2/5
Entropy_sunny = -sunny_no*m.log(sunny_no,2) - sunny_yes*m.log(sunny_yes,2)
print("Entropy_sunny",Entropy_sunny)
rain_yes = 2/5
rain_no = 3/5
Entropy_rain = -rain_no*m.log(rain_no,2) - rain_yes*m.log(rain_yes,2)
print("Entropy_Rain",Entropy_rain)
Entropy_overcast = 0
print("Entropy_overcast",Entropy_overcast)

