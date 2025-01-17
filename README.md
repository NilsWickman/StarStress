# StarStress

Projekt för examensarbete i Kognitionsvetenskap 15 HP. Ett typiskt neuralt nätverk som tränats på StarCraft 2. Majoriteten av spelmiljön och träningsflödet är inspirerat av Michał Opanowicz som utvecklade denna miljö för att möjlliggöra träning på konsument-gradig hårdvara i sitt examensarbete. https://github.com/MichalOp/StarTrain

Koden var lite gammal och behövde uppdateras i så gott som varje fil då olika bibliotek inte längre var tillgängliga.

Men det bidraget som var syftet med denna var en ny version av Transformer modulen. Detta arbete visas primärt i "self_attention_forward.py" och "StressEncoder.py" som är ersättningar av Transformer komponenter. Dessa syftar till att uppskatta resurser och hot för bättre beslutsfattning. Resultatet var en mer försiktig agent. Resultaten och jämförelse med vanlig Transformer detaljeras i mitt kandidatarbete. https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1885498&dswid=7394
