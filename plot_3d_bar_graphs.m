close all
cm1 = importdata([pwd '\data\fusion_cm_1.txt']);
cm2 = importdata([pwd '\data\fusion_cm_2.txt']);

width = 0.9;
figure
bar3(cm1,width)
xlabel('Predicted Labels')
ylabel('True Labels')
saveas(gcf,[pwd '\figures\fusion\fusion_cm_1_front.png'],'png')

figure
bar3(cm2,width)
xlabel('Predicted Labels')
ylabel('True Labels')
saveas(gcf,[pwd '\figures\fusion\fusion_cm_2_front.png'],'png')

