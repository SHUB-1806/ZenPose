Yoga, an ancient discipline emphasizing balance, flexibility,
and mindfulness, relies heavily on maintaining correct pos-
tures for achieving physiological and psychological bene-
fits. However, for most practitioners, accurately performing
these poses without expert supervision remains challenging.
With the advancement of computer vision and deep learn-
ing, automatic yoga pose recognition has become an emerg-
ing area of research aimed at assisting practitioners through
intelligent feedback systems. Traditional approaches pri-
marily depend on convolutional neural networks (CNNs)
trained directly on image pixels, which, while effective,
require large datasets and are computationally expensive.
Moreover, such models are often sensitive to background
variations, lighting conditions, and camera angles.
To overcome these challenges, this work presents a
lightweight and interpretable method for yoga pose classifi-
cation using MediaPipe Pose for keypoint extraction and a
deep neural network (DNN) trained on derived joint-angle
features. By representing each posture through geometrical
relationships among body landmarks instead of raw images,
the proposed model achieves higher generalization across
different environments while maintaining real-time perfor-
mance. This approach not only reduces computational com-
plexity but also provides a foundation for developing inter-
active yoga assistants capable of guiding users toward cor-
rect pose alignment. The novelty of this work lies in the
use of geometric, angle-based pose representation derived
from human body landmarks, instead of raw pixel data used
in CNN-based methods. This approach makes the model
lightweight, interpretable, and background-invariant while
maintaining a high accuracy of 94%. Furthermore, the pro-
posed architecture achieves real-time performance on CPU
hardware, making it suitable for low-resource yoga assis-
tance and corrective feedback systems.

## 📄 Project Report (ZenPose)

You can download the full project report here:

[📘 Download ZenPose Report (PDF)](./ZenPose.pdf)
