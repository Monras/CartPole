# KEX
Cartpole main  = koden som kör programmet 

gym_kod = gym koden med våra fixade variabler och villkor

Att göra/testa: 
 - hur många lager/neuroner behövs (testa!)
 - Ändra learning rate och keep_prob?
 - läs på mer om hur tflearn och tensorflow funkar
 - Ta fram bra träningsdata, kör lång tid med högre score_limit
 - Ta fram en bra modell, kanske olika  modeller för olika vinklar

tips: 
- Öka tidstegen när simulationen körs
- Filerna blir enorma om score_limit är för låg och vinkeln för stor
 

Model mappen
- innehåller alla modeler som är bra, sista delen i namnet borde innehålla average points.


## Gym's repository
Cartpole simulation
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
