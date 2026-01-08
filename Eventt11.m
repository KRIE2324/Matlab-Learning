clc;
clear;
clc;
clear;
x11{1}=0.3461;
x12{1}=0.2548;

x21{1}=0.1439;
x22{1}=0.4060;

x31{1}=0.1545;
x32{1}=0.2242;

x41{1}=0.1439;
x42{1}=0.3060;

x51{1}=0.1545;
x52{1}=0.2242;%智能体状态初值

x01{1}=-0.5;
x02{1}=0.5;%领导者状态初值

A=[0,1,0,0,1
   1,0,0,1,0
   0,0,0,1,1
   0,1,1,0,1
   1,0,1,1,0];%weighted adjacency matrix

B=[0,0,0,0,0
   0,0,0,0,0
   0,0,1,0,0
   0,0,0,0,0
   0,0,0,0,0];%pining gain matrix

D=[2,0,0,0,0
   0,2,0,0,0
   0,0,2,0,0
   0,0,0,3,0
   0,0,0,0,3];%degree matrix

L=D-A; %Lapalace matrix
 
Q=[1,0
   0,1];%performance function中正定矩阵，对于所有智能体均取为2阶单位阵
%  lamulaQ=1;  
 M=1;%Lipschitz constant
 Rii1=0.1;
 Rii2=0.2;
 Rii3=0.3;
 Rii4=0.4;
 Rii5=0.5;
 
R=[1,0.1,0,0,0.1
   0.1,1,0,0.1,0
   0,0,1,0.1,0.1
   0,0.1,0.1,1,0.1
   0.1,0,0.1,0.1,1];%performance function中uRu的系数

a1=0.1;
a2=0.2;
a3=0.3;
a4=0.4;
a5=0.5;%神经网络学习率

 hatW1{1}=rand(2,1);
 hatW2{1}=rand(2,1);
 hatW3{1}=rand(2,1);
 hatW4{1}=rand(2,1);
 hatW5{1}=rand(2,1);%执行网络中隐含层到输出层间权值初值
 

 step=0.01;

     
rt1=[];
rr1=[];
rt2=[];
rr2=[];
rt3=[];
rr3=[];
rt4=[];
rr4=[];
rt5=[];
rr5=[];


count1=0.01;
count2=0.01;
count3=0.01;
count4=0.01;
count5=0.01;
 




 for t=1:1000000
     time(t)=(t-1)*step;
     
     x1{t}=[x11{t};x12{t}];
     x2{t}=[x21{t};x22{t}];
     x3{t}=[x31{t};x32{t}];
     x4{t}=[x41{t};x42{t}];
     x5{t}=[x51{t};x52{t}];
     x0{t}=[x01{t};x02{t}];
     
     sigma11{t}=A(1,2)*(x11{t}-x21{t})+A(1,5)*(x11{t}-x51{t});
     sigma12{t}=A(1,2)*(x12{t}-x22{t})+A(1,5)*(x12{t}-x52{t});
     sigma21{t}=A(2,1)*(x21{t}-x11{t})+A(2,4)*(x21{t}-x41{t});
     sigma22{t}=A(2,1)*(x22{t}-x12{t})+A(2,4)*(x22{t}-x42{t});
     sigma31{t}=A(3,4)*(x31{t}-x41{t})+A(3,5)*(x31{t}-x51{t})+B(3,3)*(x31{t}-x01{t});
     sigma32{t}=A(3,4)*(x32{t}-x42{t})+A(3,5)*(x32{t}-x52{t})+B(3,3)*(x32{t}-x02{t});
     sigma41{t}=A(4,2)*(x41{t}-x21{t})+A(4,3)*(x41{t}-x31{t})+A(4,5)*(x41{t}-x51{t});
     sigma42{t}=A(4,2)*(x42{t}-x22{t})+A(4,3)*(x42{t}-x32{t})+A(4,5)*(x42{t}-x52{t});
     sigma51{t}=A(5,1)*(x51{t}-x11{t})+A(5,3)*(x51{t}-x31{t})+A(5,4)*(x51{t}-x41{t});
     sigma52{t}=A(5,1)*(x52{t}-x12{t})+A(5,3)*(x52{t}-x32{t})+A(5,4)*(x52{t}-x42{t});%local neighbor error
     
     sigma1{t}=[sigma11{t};sigma12{t}];
     sigma2{t}=[sigma21{t};sigma22{t}];
     sigma3{t}=[sigma31{t};sigma32{t}];
     sigma4{t}=[sigma41{t};sigma42{t}];
     sigma5{t}=[sigma51{t};sigma52{t}];
     
     hatsigma11{1}=sigma11{1};
     hatsigma12{1}=sigma12{1};
     hatsigma21{1}=sigma21{1};
     hatsigma22{1}=sigma22{1};
     hatsigma31{1}=sigma31{1};
     hatsigma32{1}=sigma32{1};
     hatsigma41{1}=sigma41{1};
     hatsigma42{1}=sigma42{1};
     hatsigma51{1}=sigma51{1};
     hatsigma52{1}=sigma52{1};
     
     hatsigma1{t}=[hatsigma11{t};hatsigma12{t}];
     hatsigma2{t}=[hatsigma21{t};hatsigma22{t}];
     hatsigma3{t}=[hatsigma31{t};hatsigma32{t}];
     hatsigma4{t}=[hatsigma41{t};hatsigma42{t}];
     hatsigma5{t}=[hatsigma51{t};hatsigma52{t}];
     
     e1{t}=hatsigma1{t}-sigma1{t};
     e2{t}=hatsigma2{t}-sigma2{t};
     e3{t}=hatsigma3{t}-sigma3{t};
     e4{t}=hatsigma4{t}-sigma4{t};
     e5{t}=hatsigma5{t}-sigma5{t};
     
     fai1{t}=tanh(sigma1{t});
     fai2{t}=tanh(sigma2{t});
     fai3{t}=tanh(sigma3{t});
     fai4{t}=tanh(sigma4{t});
     fai5{t}=tanh(sigma5{t});
     
     tidufai1{t}=[1-tanh(sigma11{t})^2,0
                  0,1-tanh(sigma12{t})^2];
              
     tidufai2{t}=[1-tanh(sigma21{t})^2,0
                  0,1-tanh(sigma22{t})^2];
     
     tidufai3{t}=[1-tanh(sigma31{t})^2,0
                  0,1-tanh(sigma32{t})^2];
              
     tidufai4{t}=[1-tanh(sigma41{t})^2,0
                  0,1-tanh(sigma42{t})^2];
              
     tidufai5{t}=[1-tanh(sigma51{t})^2,0
                  0,1-tanh(sigma52{t})^2];
              
     ttidufai1{t}=[1-tanh(hatsigma11{t})^2,0
                  0,1-tanh(hatsigma12{t})^2];
              
     ttidufai2{t}=[1-tanh(hatsigma21{t})^2,0
                  0,1-tanh(hatsigma22{t})^2];
     
     ttidufai3{t}=[1-tanh(hatsigma31{t})^2,0
                  0,1-tanh(hatsigma32{t})^2];
              
     ttidufai4{t}=[1-tanh(hatsigma41{t})^2,0
                  0,1-tanh(hatsigma42{t})^2];
              
     ttidufai5{t}=[1-tanh(hatsigma51{t})^2,0
                  0,1-tanh(hatsigma52{t})^2];
              
              
     f1{t}=[x12{t}-x11{t}^2*x12{t};-(x11{t}+x12{t})*(1-x11{t})^2];
     f2{t}=[x22{t}-x21{t}^2*x22{t};-(x21{t}+x22{t})*(1-x21{t})^2];
     f3{t}=[x32{t}-x31{t}^2*x32{t};-(x31{t}+x32{t})*(1-x31{t})^2];
     f4{t}=[x42{t}-x41{t}^2*x42{t};-(x41{t}+x42{t})*(1-x41{t})^2];
     f5{t}=[x52{t}-x51{t}^2*x52{t};-(x51{t}+x52{t})*(1-x51{t})^2];
     f0{t}=[x02{t}-x01{t}^2*x02{t};-(x01{t}+x02{t})*(1-x01{t})^2];
     
     g1{t}=[0;x12{t}];
     g2{t}=[0;x22{t}];
     g3{t}=[0;x32{t}];
     g4{t}=[0;x42{t}];
     g5{t}=[0;x52{t}];
     
     tidef1{t}=f1{t}-f0{t};
     tidef2{t}=f2{t}-f0{t};
     tidef3{t}=f3{t}-f0{t};
     tidef4{t}=f4{t}-f0{t};
     tidef5{t}=f5{t}-f0{t};
     
     u1{t}=-(L(1,1)+B(1,1))/2*inv(R(1,1))*g1{t}'*ttidufai1{t}'*hatW1{t};
     u2{t}=-(L(2,2)+B(2,2))/2*inv(R(2,2))*g2{t}'*ttidufai2{t}'*hatW2{t};
     u3{t}=-(L(3,3)+B(3,3))/2*inv(R(3,3))*g3{t}'*ttidufai3{t}'*hatW3{t};
     u4{t}=-(L(4,4)+B(4,4))/2*inv(R(4,4))*g4{t}'*ttidufai4{t}'*hatW4{t};
     u5{t}=-(L(5,5)+B(5,5))/2*inv(R(5,5))*g5{t}'*ttidufai5{t}'*hatW5{t};
     
     u11{t}=-(L(1,1)+B(1,1))/2*inv(R(1,1))*g1{t}'*tidufai1{t}'*hatW1{t};
     u12{t}=-(L(2,2)+B(2,2))/2*inv(R(2,2))*g2{t}'*tidufai2{t}'*hatW2{t};
     u13{t}=-(L(3,3)+B(3,3))/2*inv(R(3,3))*g3{t}'*tidufai3{t}'*hatW3{t};
     u14{t}=-(L(4,4)+B(4,4))/2*inv(R(4,4))*g4{t}'*tidufai4{t}'*hatW4{t};
     u15{t}=-(L(5,5)+B(5,5))/2*inv(R(5,5))*g5{t}'*tidufai5{t}'*hatW5{t};
     
     
     
     U1{t}=sigma1{t}'*Q*sigma1{t}+u1{t}*R(1,1)*u1{t}+u2{t}*R(1,2)*u2{t}+u5{t}*R(1,5)*u5{t};
     U2{t}=sigma2{t}'*Q*sigma2{t}+u2{t}*R(2,2)*u2{t}+u1{t}*R(2,1)*u1{t}+u4{t}*R(2,4)*u4{t};
     U3{t}=sigma3{t}'*Q*sigma3{t}+u3{t}*R(3,3)*u3{t}+u4{t}*R(3,4)*u4{t}+u5{t}*R(3,5)*u5{t};
     U4{t}=sigma4{t}'*Q*sigma4{t}+u4{t}*R(4,4)*u4{t}+u2{t}*R(4,2)*u2{t}+u3{t}*R(4,3)*u3{t}+u5{t}*R(4,5)*u5{t};
     U5{t}=sigma5{t}'*Q*sigma5{t}+u5{t}*R(5,5)*u5{t}+u1{t}*R(5,1)*u5{t}+u3{t}*R(5,3)*u3{t}+u4{t}*R(5,4)*u4{t};
     
     delta1{t}=tidufai1{t}*(L(1,2)*(tidef2{t}+g2{t}*u2{t})+L(1,5)*(tidef5{t}+g5{t}*u5{t}));
     delta2{t}=tidufai2{t}*(L(2,1)*(tidef1{t}+g1{t}*u1{t})+L(2,4)*(tidef4{t}+g4{t}*u4{t}));
     delta3{t}=tidufai3{t}*(L(3,4)*(tidef4{t}+g4{t}*u4{t})+L(3,5)*(tidef5{t}+g5{t}*u5{t}));
     delta4{t}=tidufai4{t}*(L(4,2)*(tidef2{t}+g2{t}*u2{t})+L(4,3)*(tidef3{t}+g3{t}*u3{t})+L(4,5)*(tidef5{t}+g5{t}*u5{t}));
     delta5{t}=tidufai5{t}*(L(5,1)*(tidef1{t}+g1{t}*u1{t})+L(5,3)*(tidef3{t}+g3{t}*u3{t})+L(5,4)*(tidef4{t}+g4{t}*u4{t}));

%        norm(e1{t})<=sqrtm(((1-(tao1)^2)*(norm(sigma1{t}))^2)/(((1+(epsilon1)^2)*(M^2)*Rii))); 
        if norm(e1{t})<=sqrtm(((norm(sigma1{t}))^2)/((2*(M^2)*Rii1))); 
         hatsigma11{t+1}=hatsigma11{t};
         hatsigma12{t+1}=hatsigma12{t};
%          hatW1{t+1}=hatW1{t};
     else
         rt1=[rt1;t*step]; %触发次数                   
         rr1=[rr1;t*step-count1];   %触发间隔              
         count1=t*step;
         hatsigma11{t+1}=sigma11{t};
         hatsigma12{t+1}=sigma12{t};
%          hatW1{t+1}=hatW1{t}-a1*delta1{t}*(delta1{t}'*hatW1{t}+U1{t});
     end
         
     if norm(e2{t})<=sqrtm(((norm(sigma2{t}))^2)/((2*(M^2)*Rii2))); 
         hatsigma21{t+1}=hatsigma21{t};
         hatsigma22{t+1}=hatsigma22{t};
     else
         rt2=[rt2;t*step];                       
         rr2=[rr2;t*step-count2];               
         count2=t*step;
         hatsigma21{t+1}=sigma21{t};
         hatsigma22{t+1}=sigma22{t};
     end
         
     if norm(e3{t})<=sqrtm(((norm(sigma3{t}))^2)/((2*(M^2)*Rii3))); 
         hatsigma31{t+1}=hatsigma31{t};
         hatsigma32{t+1}=hatsigma32{t};
     else
         rt3=[rt3;t*step];                    
         rr3=[rr3;t*step-count3];               
         count3=t*step;
         hatsigma31{t+1}=sigma31{t};
         hatsigma32{t+1}=sigma32{t};
     end
         
     if norm(e4{t})<=sqrtm(((norm(sigma4{t}))^2)/((2*(M^2)*Rii4))); 
         hatsigma41{t+1}=hatsigma41{t};
         hatsigma42{t+1}=hatsigma42{t};
     else
         rt4=[rt4;t*step];                     
         rr4=[rr4;t*step-count4];             
         count4=t*step;
         hatsigma41{t+1}=sigma41{t};
         hatsigma42{t+1}=sigma42{t};
     end
         
     if norm(e5{t})<=sqrtm(((norm(sigma5{t}))^2)/((2*(M^2)*Rii5))); 
         hatsigma51{t+1}=hatsigma51{t};
         hatsigma52{t+1}=hatsigma52{t};
     else
         rt5=[rt5;t*step];                      
         rr5=[rr5;t*step-count5]; 
         count5=t*step;
         hatsigma51{t+1}=sigma51{t};
         hatsigma52{t+1}=sigma52{t};
     end
     
     
     
     hatW1{t+1}=hatW1{t}-a1*delta1{t}*(delta1{t}'*hatW1{t}+U1{t});
     hatW2{t+1}=hatW2{t}-a2*delta2{t}*(delta2{t}'*hatW2{t}+U2{t});
     hatW3{t+1}=hatW3{t}-a3*delta3{t}*(delta3{t}'*hatW3{t}+U3{t});
     hatW4{t+1}=hatW4{t}-a4*delta4{t}*(delta4{t}'*hatW4{t}+U4{t});
     hatW5{t+1}=hatW5{t}-a5*delta5{t}*(delta5{t}'*hatW5{t}+U5{t});
     
     x11{t+1}= x11{t}+(x12{t}-x11{t}^2*x12{t})*step;
     x12{t+1}= x12{t}+(-(x11{t}+x12{t})*(1-x11{t})^2+x12{t}*u1{t})*step;
     
     x21{t+1}= x21{t}+(x22{t}-x21{t}^2*x22{t})*step;
     x22{t+1}= x22{t}+(-(x21{t}+x22{t})*(1-x21{t})^2+x22{t}*u2{t})*step;
     
     x31{t+1}= x31{t}+(x32{t}-x31{t}^2*x32{t})*step;
     x32{t+1}= x32{t}+(-(x31{t}+x32{t})*(1-x31{t})^2+x32{t}*u3{t})*step;
   
     x41{t+1}= x41{t}+(x42{t}-x41{t}^2*x42{t})*step;
     x42{t+1}= x42{t}+(-(x41{t}+x42{t})*(1-x41{t})^2+x42{t}*u4{t})*step;
     
     x51{t+1}= x51{t}+(x52{t}-x51{t}^2*x52{t})*step;
     x52{t+1}= x52{t}+(-(x51{t}+x52{t})*(1-x51{t})^2+x52{t}*u5{t})*step;
     
     x01{t+1}= x01{t}+(x02{t}-x01{t}^2*x02{t})*step;
     x02{t+1}= x02{t}+(-(x01{t}+x02{t})*(1-x01{t})^2)*step;%系统方程
     
     
     qq(t)=x01{t};
     ww(t)=x11{t};
     ee(t)=x21{t};
     rr(t)=x31{t};
     tt(t)=x41{t};
     yy(t)=x51{t};
     uu(t)=x02{t};
     ii(t)=x12{t};
     oo(t)=x22{t};
     pp(t)=x32{t};
     aa(t)=x42{t};
     ss(t)=x52{t};
     
     quan1(t)=norm(hatW1{t});
     quan2(t)=norm(hatW2{t});
     quan3(t)=norm(hatW3{t});
     quan4(t)=norm(hatW4{t});
     quan5(t)=norm(hatW5{t});
     
     
     vvv1(t)=u1{t};
     vvv2(t)=u2{t};
     vvv3(t)=u3{t};
     vvv4(t)=u4{t};
     vvv5(t)=u5{t};
     
     vvv11(t)=u11{t};
     vvv12(t)=u12{t};
     vvv13(t)=u13{t};
     vvv14(t)=u14{t};
     vvv15(t)=u15{t};
     
     
     UUUU1(t)=U1{t};
     UUUU2(t)=U2{t};
     UUUU3(t)=U3{t};
     UUUU4(t)=U4{t};
     UUUU5(t)=U5{t};
        
     
     
     yuzhi1(t)=(norm(e1{t}));
     the1(t)=sqrtm(((norm(sigma1{t}))^2)/((2*(M^2)*Rii1)));
     
     yuzhi2(t)=(norm(e2{t}));
     the2(t)=sqrtm(((norm(sigma2{t}))^2)/((2*(M^2)*Rii2)));
     
     yuzhi3(t)=(norm(e3{t}));
     the3(t)=sqrtm(((norm(sigma3{t}))^2)/((2*(M^2)*Rii3)));
     
     yuzhi4(t)=(norm(e4{t}));
     the4(t)=sqrtm(((norm(sigma4{t}))^2)/((2*(M^2)*Rii4)));
     
     yuzhi5(t)=(norm(e5{t}));
     the5(t)=sqrtm(((norm(sigma5{t}))^2)/((2*(M^2)*Rii5)));
     
     if t*step>30
         break; 
     end
 end


 
 figure(1)
plot(time,qq,'k',time,ww,'r',time,ee,'m',time,rr,'c',time,tt,'g',time,yy,'-.b','LineWidth',1.5)
set(gca, 'FontSize', 11)
legend('x_{01}','x_{11}','x_{21}','x_{31}','x_{41}','x_{51}');
xlabel('Time(Sec)')  

 figure(2)
plot(time,uu,'m',time,ii,'b',time,oo,'r',time,pp,'-.k',time,aa,'g',time,ss,'c','LineWidth',1.5)
set(gca, 'FontSize', 11)
legend('x_{02}','x_{12}','x_{22}','x_{32}','x_{42}','x_{52}');
xlabel('Time(Sec)')  

 
figure(3)
% plot(time,quan1,'b',time,quan2,'g',time,quan3,'r',time,quan4,'k',time,quan5,'m','LineWidth',1.5)
plot(time(1:100:end), quan1(1:100:end),'--s',time(1:100:end), quan2(1:100:end),'--+',time(1:100:end), quan3(1:100:end),'--*',time(1:100:end), quan4(1:100:end),'--*',time(1:100:end), quan5(1:100:end),'--*','linewidth', 1.5)
set(gca, 'FontSize', 11)
legend('$\hat{W}_{c1}$','$\hat{W}_{c2}$','$\hat{W}_{c3}$','$\hat{W}_{c4}$','$\hat{W}_{c5}$')
xl=xlabel('Time(Sec)');
set(xl,'Interpreter','latex')
ll=legend('$\hat{W}_{c1}$','$\hat{W}_{c2}$','$\hat{W}_{c3}$','$\hat{W}_{c4}$','$\hat{W}_{c5}$');
set(ll,'Interpreter','latex');



figure(4)
subplot(5,1,1)
stem(rt1,rr1)
xlabel('Time(sec)')
ylabel('Agent 1')
h = legend('Agent 1');
set(gca, 'FontSize', 11)
set(h,'Interpreter','latex')


subplot(5,1,2)
stem(rt2,rr2)
xlabel('Time(sec)')
ylabel('Agent 2')
set(gca, 'FontSize', 11)
h = legend('Agent 2');
set(h,'Interpreter','latex')


subplot(5,1,3)
stem(rt2,rr2)
xlabel('Time(sec)')
ylabel('Agent 3')
set(gca, 'FontSize', 11)
h = legend('Agent 3');
set(h,'Interpreter','latex')


subplot(5,1,4)
stem(rt2,rr2)
xlabel('Time(sec)')
ylabel('Agent 4')
set(gca, 'FontSize', 11)
h = legend('Agent 4');
set(h,'Interpreter','latex')


subplot(5,1,5)
stem(rt2,rr2)
xlabel('Time(sec)')
ylabel('Agent 5')
set(gca, 'FontSize', 11)
h = legend('Agent 5');
set(h,'Interpreter','latex')





figure(5)
 plot(time,yuzhi1,'k',time,the1,'r','LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('||{{e_t1}}||','||{{e_T1}}||');
 xlabel('Time(Sec)') 

figure(6)
 plot(time,yuzhi2,'k',time,the2,'r','LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('||{{e_t2}}||','||{{e_T2}}||');
 xlabel('Time(Sec)') 

figure(7)
 plot(time,yuzhi3,'k',time,the3,'r','LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('||{{e_t3}}||','||{{e_T3}}||');
 xlabel('Time(Sec)') 


 figure(8)
 plot(time,yuzhi4,'k',time,the4,'r','LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('||{{e_t4}}||','||{{e_T4}}||');
 xlabel('Time(Sec)') 

figure(9)
 plot(time,yuzhi5,'k',time,the5,'r','LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('||{{e_t5}}||','||{{e_T5}}||');
 xlabel('Time(Sec)') 


%shi
     U1_sum(1)=0;
 for i=1:1:300
         U1_sum(i+1)=U1_sum(i)+ UUUU1(i);
 end
 
    U2_sum(1)=0;
 for i=1:1:300
         U2_sum(i+1)=U2_sum(i)+ UUUU2(i);
 end

   U3_sum(1)=0;
 for i=1:1:300
         U3_sum(i+1)=U3_sum(i)+ UUUU3(i);
 end

  U4_sum(1)=0;
 for i=1:1:300
         U4_sum(i+1)=U4_sum(i)+ UUUU4(i);
 end
 
 U5_sum(1)=0;
 for i=1:1:300
         U5_sum(i+1)=U5_sum(i)+ UUUU5(i);
 end
 
%   figure(19)
% plot(time,U1_sum,'k',time,U2_sum,'r',time,U3_sum,'m',time,U4_sum,'c',time,U5_sum,'g')
% legend('J1','J2','J3','J4','J5');
% xlabel('Time(Sec)')  

% $\hat{u}_{5}^{\ast }$
% figure(10)
% plot(time,vvv11,'k',time,vvv12,'b',time,vvv13,'m',time,vvv14,'r',time,vvv15,'g','LineWidth',1.5)
% set(gca, 'FontSize', 11)
% legend('$\overline {u}_{1}$','$\overline {u}_{2}$','$\overline {u}_{3}$','$\overline {u}_{4}$','$\overline {u}_{5}$')
% xl=xlabel('Time(Sec)');
% set(xl,'Interpreter','latex')
% ll=legend('$\overline {u}_{1}$','$\overline {u}_{2}$','$\overline {u}_{3}$','$\overline {u}_{4}$','$\overline {u}_{5}$');
% set(ll,'Interpreter','latex','fontsize',14,'FontName','Times New Roman','fontweight','bold')
% set(ll,'Interpreter','latex');
% 
figure(11)
plot(time,vvv1,'k',time,vvv2,'b',time,vvv3,'m',time,vvv4,'r',time,vvv5,'g','LineWidth',1.5)
legend('Agent 1','Agent 2','Agent 3','Agent 4','Agent 5')
set(gca, 'FontSize', 11)
xl=xlabel('Time(Sec)');
set(xl,'Interpreter','latex')
ll=legend('Agent 1','Agent 2','Agent 3','Agent 4','Agent 5');
set(ll,'Interpreter','latex','fontsize',14,'FontName','Times New Roman','fontweight','bold')
set(ll,'Interpreter','latex');




%%%%%%%%%%%%%%%%%%%               不能同时运行                 %%%%%%%%%%%%%
%  oo=1:3001;
%  pp=1:16;
%  figure(9)
%  plot(time,oo,'-.',rt1,pp,'LineWidth',1.5)
%  set(gca, 'FontSize', 11)
%  legend('ADP method of agent 1','ETADP  method of agent 1');
%  xlabel('Time(Sec)') 

 
%  zz=1:3001;
%  mm=1:23;
%  figure(10)
%  plot(time,zz,'-.',rt2,mm,'LineWidth',1.5)
% set(gca, 'FontSize', 11)
%  legend('ADP method of agent 2','ETADP  method of agent 2');
%  xlabel('Time(Sec)') 


% 
%  bb=1:3001;
%  cc=1:30;
%  figure(11)
%  plot(time,bb,'-.',rt3,cc,'LineWidth',1.5)
%  set(gca, 'FontSize', 11)
%  legend('ADP method of agent 3','ETADP  method of agent 3');
%  xlabel('Time(Sec)') 

 
%  xx=1:3001;
%  yy=1:33;
%  figure(12)
%  plot(time,xx,'-.',rt4,yy,'LineWidth',1.5)
%  set(gca, 'FontSize', 11)
%  legend('ADP method of agent 4','ETADP  method of agent 4');
%  xlabel('Time(Sec)') 
 
 ll=1:3001;
 ee=1:33;
 figure(13)
 plot(time,ll,'-.',rt5,ee,'LineWidth',1.5)
 set(gca, 'FontSize', 11)
 legend('ADP method of agent 5','ETADP  method of agent 5');
 xlabel('Time(Sec)') 
 
 
 