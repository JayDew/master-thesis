# Define a function using lambda
import numpy as np
from matplotlib import pyplot as plt

stock = lambda A, amp, angle, phase: A * angle + amp * np.sin(angle + phase)


# on a small instance 2X2 default one !!
CKKS = [0.49968673807281583, 0.4993120374847235, 0.498300356658286, 0.497589087261864, 0.4971227616903797, 0.49645132667673453, 0.4943852123446817, 0.49395286471618105, 0.49335744931522263, 0.4917807955084883, 0.49159065577587957, 0.4895146119879987, 0.4896562170889043, 0.4884129849405565, 0.487837721833867, 0.48703390452969186, 0.4850217630561411, 0.4854031218883714, 0.48332580175115103, 0.4830432130060278, 0.4805699391889713, 0.4819204550743399, 0.47847899171516994, 0.4796480373024581, 0.4762644376238103, 0.47857216673934144, 0.4740720387489923, 0.4767557016354287, 0.4724573536061685, 0.4732987142290177, 0.4709437067294169, 0.4717645313613059, 0.4691397659420333, 0.46875558836651265, 0.46774363621926596, 0.4664903458771712, 0.46543655891629543, 0.46459165980170014, 0.4635229169695767, 0.4613797932821222, 0.46230232057646914, 0.4583905332142628, 0.46110895111867256, 0.45512284583473184, 0.45989959695428473, 0.45317213707897785, 0.4569044576272241, 0.45178228315235147, 0.45500367857310264, 0.44956537017234943, 0.4525906041477872, 0.44798090296132953, 0.4490673275046664, 0.446828153390038, 0.4461028273880854, 0.44518186286502237, 0.44406816924966597, 0.44307232602971114, 0.44230041363669725, 0.44059570203962184, 0.44107126697571886, 0.43843747893938767, 0.4389905703114141, 0.43600843331827344, 0.437647550725635, 0.433242349570381, 0.43622430143666846, 0.4318945077940646, 0.43318725248566725, 0.42941960111314503, 0.43162805095790846, 0.42822362737972774, 0.42812707366572234, 0.4251782747848445, 0.4262829089660636, 0.4224381864486886, 0.4248496919878044, 0.4202808413506878, 0.4228532862705774, 0.4179421080247221, 0.4209766724787807, 0.41676264115678974, 0.41852671895572513, 0.4133246499102148, 0.4165157710076097, 0.4115517432371876, 0.41429409784810467, 0.4101286929240424, 0.41066547406281595, 0.4089232044900424, 0.40760325530230324, 0.40731210312531796, 0.40511215698069414, 0.4064446041550913, 0.4025989187340131, 0.40404474411186775, 0.40124004566352944, 0.40184025622089425, 0.3998112601837185, 0.39867180507880895, 0.39768818249745364, 0.397525769572721, 0.39592689410198756, 0.39595336608467413, 0.3932058759952701, 0.3942529658455829, 0.3904918846873755, 0.39354767208565145, 0.3888314488845219, 0.390867872437596, 0.38693683473495977, 0.38775020358451817, 0.38472069085787886, 0.38587383319621305, 0.38212235666745054, 0.3841879525643579, 0.3800934127717437, 0.3818774259150899, 0.37812127346620716, 0.37923048678166804, 0.3768914594522421, 0.3762605974243718, 0.3748824213140476, 0.37398341565637283, 0.3741925876744996, 0.3726553176732037, 0.3721726261612397, 0.3704777724571908, 0.369139114915924, 0.36889711514925816, 0.367239238112429, 0.36609925626990725, 0.36619203815891277, 0.36468613888738366, 0.36376261423627865, 0.36317136408917694, 0.36169801554006886, 0.36072670194651923, 0.36032572658730366, 0.35893247084545354, 0.35772883047054616, 0.3564513803515572, 0.3560358563764674, 0.35307611934187166, 0.35506966242143, 0.35181182142206424, 0.3529207989686287, 0.34944135467248794, 0.3503869542284139, 0.3481511863997908, 0.3485864618118688, 0.34504117379869514, 0.3471499168563965, 0.3428844241685658, 0.3451882139972078, 0.3403559337550326, 0.34363547018698015, 0.33881140702352575, 0.3409918792769224, 0.3367585605932968, 0.33947801094507835, 0.3344737542471121, 0.337814801170782, 0.3322582871933395, 0.33556731800704886, 0.3307071860495667, 0.3322742967355161, 0.32944404520045456, 0.3299541833307016, 0.32832026828453387, 0.32752314225855583, 0.3255305024707551, 0.3265525168925252, 0.32219295710296514, 0.3253441061993403, 0.3194956586528088, 0.32297954245647276, 0.3177351104310966, 0.3211013665719731, 0.31482925970805, 0.31925783980455424, 0.3128129560747981, 0.31684418566413886, 0.3106894289800808, 0.31439443199437855, 0.30929609780763356, 0.3119018534054554, 0.3067826817780377, 0.3097312368306002, 0.30374311478345994, 0.30879029791541646, 0.3017892390992478, 0.30692969269473913, 0.29868217849131207, 0.3041497429796598, 0.2979709012556093, 0.3009439049745555, 0.29518870969074174, 0.3003049590336495, 0.29090667579118956]
BFV = [0.49949999898672104, 0.4990000054240227, 0.4983750060200691, 0.4976250007748604, 0.49674998968839645, 0.49574998766183853, 0.4946249797940254, 0.4933749660849571, 0.49199996143579483, 0.49049995094537735, 0.4888749346137047, 0.487124927341938, 0.48524991422891617, 0.48324989527463913, 0.4811248853802681, 0.4788748696446419, 0.47649984806776047, 0.47399983555078506, 0.4713748171925545, 0.4686247929930687, 0.4657497778534889, 0.46274975687265396, 0.4596247300505638, 0.45637471228837967, 0.45299968868494034, 0.4494996592402458, 0.4458746388554573, 0.4421246126294136, 0.4382495805621147, 0.43424955755472183, 0.43012452870607376, 0.4258744940161705, 0.42149946838617325, 0.4169994369149208, 0.4123743996024132, 0.40762437134981155, 0.40274933725595474, 0.39774929732084274, 0.39262426644563675, 0.38737422972917557, 0.3819991871714592, 0.37649915367364883, 0.3708741143345833, 0.36512406915426254, 0.3592490330338478, 0.3532489910721779, 0.3471239432692528, 0.3408739045262337, 0.3344988599419594, 0.3279988095164299, 0.3213737681508064, 0.31462372094392776, 0.3077486678957939, 0.30074862390756607, 0.29362357407808304, 0.2863735184073448, 0.2789984717965126, 0.2714984193444252, 0.2638733610510826, 0.256123311817646, 0.24824825674295425, 0.2402481958270073, 0.23212314397096634, 0.2238730862736702, 0.21549802273511887, 0.20699796825647354, 0.19837290793657303, 0.18962284177541733, 0.18074778467416763, 0.17174772173166275, 0.16262265294790268, 0.15337259322404861, 0.14399752765893936, 0.13449745625257492, 0.12487239390611649, 0.11512232571840286, 0.10524725168943405, 0.09524718672037125, 0.08512211591005325, 0.07487203925848007, 0.0644969716668129, 0.05399689823389053, 0.04337181895971298, 0.03262174874544144, 0.021746672689914703, 0.010746590793132782, 0.00037848204374313354, 0.01125645637512207, 0.011385917663574219, 0.0062555596232414246, 0.0011462494730949402, 0.001399122178554535, 0.0028084218502044678, 0.0036732181906700134, 0.004010982811450958, 0.0038382932543754578, 0.003170885145664215, 0.0020237043499946594, 0.00041093677282333374, 0.0016539320349693298, 0.002528458833694458, 0.002188757061958313, 0.001429431140422821, 0.0008459016680717468, 0.0006394460797309875, 0.0007193908095359802, 0.0008985549211502075, 0.0010363981127738953, 0.0010852441191673279, 0.0010663866996765137, 0.0010240152478218079, 0.0009913817048072815, 0.0009798035025596619, 0.0009842664003372192]


x = np.arange(max(len(BFV), len(CKKS)))  # x-axis

# Create main container with size of 6x5
fig = plt.figure(figsize=(6, 5))

# Create first axes, the top-left plot with green plot
# sub1 = fig.add_subplot(2, 2, 1)  # two rows, two columns, fist cell
# sub1.plot(np.arange(len(fista)), fista, color='red')
# sub1.plot(np.arange(len(pgd)), pgd, color='blue')
# sub1.set_xlim(200, 245)
# sub1.set_ylim(0, .4)
# sub1.set_ylabel(r'$ \frac{|f(x_k) - f^{*}|}{|f^{*}}$', labelpad=15)

# Create second axes, a combination of third and fourth cell
# sub2 = fig.add_subplot(2,2,2) # two rows, two columns, second cell
# sub2.plot(np.arange(len(fista)), fista, color='red')
# sub2.plot(np.arange(len(pgd)), pgd, color='blue')
# sub2.set_xlim(350, 2500)
# sub2.set_ylim(-.06, .01)
# sub2.set_ylabel(r'$ \frac{|f(x_k) - f^{*}|}{|f^{*}}$', labelpad=15)

# Create third axes, a combination of third and fourth cell
sub3 = fig.add_subplot(2, 2, (3, 4))  # two rows, two colums, combined third and fourth cell
sub3.plot(np.arange(len(BFV)), BFV, color='blue', alpha=.7, label='BFV')
sub3.plot(np.arange(len(CKKS)), CKKS, color='red', alpha=.7, label='CKKS')
sub3.set_xlim(0, 200)
sub3.set_ylim(0, 0.8)
sub3.set_xlabel(r'k-th iteration', labelpad=15)
sub3.set_ylabel(r'$ \frac{|f(x_k) - f^{*}|}{|f^{*}|}$', labelpad=15)

# Create blocked area in third axes
# sub3.fill_between((200, 245), 0, 34, facecolor='blue', alpha=0.2)  # blocked area for first axes
# sub3.fill_between((350, 2500), 0, 34, facecolor='blue', alpha=0.2)  # blocked area for first axes

sub3.legend()
# Save figure with nice margin
plt.savefig('bfv_vs_ckks.png', dpi=300, bbox_inches='tight', pad_inches=.1)
