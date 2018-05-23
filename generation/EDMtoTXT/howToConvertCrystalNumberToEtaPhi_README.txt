// CRYSTALNUMBER goes from 0 to 61199
// IETA goes from -85 to +85, and there is no zero
// IPHI goes from 1 to 360

i = CRYSTALNUMBER;
IPHI = (i+1)%360;
SIGN = i > 30599 ? 1 : -1;
if(SIGN > 0) {
    IETA = (i+1-IPHI)/360 - 84;
    }
if(SIGN < 0) {
    IETA = (i+1-IPHI)/360 - 85;
}