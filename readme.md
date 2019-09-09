# DD-Pose - A large-scale Driver Head Pose Benchmark

![DD-Pose](resources/eye-catcher.png)

Markus Roth and Dariu Gavrila  
https://dd-pose-dataset.tudelft.nl

## Contact

Please feel free to contact us with any questions, suggestions or comments:

Markus Roth  
Department of Cognitive Robotics  
Faculty of Mechanical, Maritime and Materials Engineering (3mE)  
Mekelweg 2, 2628 CD Delft, The Netherlands

contact-dd-pose-dataset@tudelft.nl

## Installation
    
Clone this repository:

    cd ~
    git clone https://github.com/herr-biber/dd-pose.git

Set access credentials in `dd-pose/00-activate.sh`

    DD_POSE_USER=<user from dd-pose-dataset.tudelft.nl registration>
    DD_POSE_PASSWORD=<password from dd-pose-dataset.tudelft.nl registration>

Source `dd-pose/00-activate.sh` to set environment variables for successive scripts

    source dd-pose/00-activate.sh

Setup environment.  
This needs sudo rights in order to install system dependencies.  
A python virtualenv is being created in `dd-pose/venv`.
Python dependencies are installed into the virtualenv

    bash dd-pose/00-setup-environment.sh

## Download data

The following scripts in `$DD_POSE_DIR` download the dataset, verify its integrity and extract it:
* `01-download-data.sh`: download files from https://dd-pose-dataset.tudelft.nl using the configured credentials into `$DD_POSE_DATA_ROOT_DIR/00-download`
* `02-compute-md5sums.sh`: compute md5sums of the downloaded files
* `03-compare-md5sums.sh`: compare md5sums of downloaded files with the md5sums from the files residing on the server
* `04-extract.sh`: extract downloaded files to `$DD_POSE_DATA_DIR`

## Create dashboard images and videos (optional)
A dashboard image displays all measurements and annotations which are available at a glance.
We provide the following scripts in `$DD_POSE_DIR`
* `05-create-dashboard-images.sh`: create dashboard images of all scenarios of all *trainval* subjects to `$DD_POSE_DATA_DIR/02-dashboard-images`
* `06-encode-dashboard-images.sh`: encode dashboard image files to compressed video files for all scenarios for all *trainval* subjects to `$DD_POSE_DATA_DIR/03-dashboard-videos`


## Getting started

Check the jupyter notebook `dd-pose/getting-started.ipynb`.
It guides you through the data structures.

    source dd-pose/00-activate.sh
    jupyter notebook --notebook-dir=$DD_POSE_DIR

Additionally, you can check the function `def get_dashboard()` in `$DD_POSE_DIR/dd_pose/visualization_helpers.py` which uses all provided data to create a dashboard image of this kind:  

![Example dashboard image](resources/sample-dashboard-image.png)

## Citation

When using the dataset please add the following citation:

M. Roth and D. M. Gavrila. DD-Pose - A large-scale Driver Head Pose Benchmark. 2019 IEEE Intelligent Vehicles Symposium (IV), 2019  
https://doi.org/10.1109/IVS.2019.8814103

    @inproceedings{roth2019iv, 
    author={M. Roth and D. M. Gavrila}, 
    title={DD-Pose - A large-scale Driver Head Pose Benchmark}, 
    booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)}, 
    year={2019}, 
    volume={}, 
    number={}, 
    pages={927-934}, 
    keywords={}, 
    doi={10.1109/IVS.2019.8814103}, 
    ISSN={2642-7214}, 
    month={June},}

## License agreement

### DD-POSE DATASET RESEARCH USE LICENSE

#### Definitions

* DD-Pose Dataset: all sensor and auxiliary data (e.g. annotations) pertaining to the DD-Pose Dataset, as described on its official website hosted by Delft University of Technology
* DD-Pose Test Dataset: all sensor and auxiliary data (e.g. annotations) pertaining to the sub-part of the DD-Pose Dataset dedicated to the performance evaluation of methods, as described on the official DD-Pose Dataset website hosted by Delft University of Technology
* Qualified User of the DD-Pose Dataset: an individual who is enrolled in a Master or Ph.D. program at an academic institution, or who is employed as scientific staff at an academic institution or at a non-profit research organization
* Derivatives:  modifications, adaptations, or compilations of the DD-Pose Dataset, or any other derivative works such as abstract representations (e.g. models) or annotations obtained from the DD-Pose Dataset
* "Research Purpose: use solely in conjunction with (1) internal non-published research projects and (2) publicly available research publications and studies, access to which is not subject to payment. The term "Research Purpose" excludes without limitation the creation of Derivatives or use in connection with the development of any product, service, software or technology or any commercial undertaking.
* GDPR: Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the Processing of Personal Data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation).
* Applicable Legislation and Regulations concerning the Processing of Personal Data: the applicable legislation and regulations and/or (further) treaties, regulations, directives, decrees, policy rules, instructions and/or recommendations from a competent public body concerning the Processing (see below) of Personal Data (see below), also including future amendments of and/or supplements thereto, including laws of the Member States implementing the GDPR and the Telecommunications Act.
* Data Subject: the identified or identifiable natural person to whom the Personal Data pertain, as referred to in Article 4 at 1) GDPR.
* Personal Data: all information relating to a Data Subject; a natural person who can be directly or indirectly identified, in particular based on an identifier such as a name, an identification number, an online identifier or one or more elements that are characteristic of the physical, physiological, genetic, psychological, economic, cultural or social identity of that natural person, as referred to in Article 4 at 1) GDPR, is deemed identifiable.
* Personal Data Breach: (suspicion of) a breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorized disclosure of, or access to, Personal Data transmitted, stored or otherwise processed, as referred to in Article 4 at 12) GDPR.
* Processing: any operation or set of operations which is performed on Personal Data or on sets of Personal Data, whether or not by automated means, such as collection, recording, organization, structuring, storage, adaptation or alteration, retrieval, consultation, use, disclosure by transmission, dissemination or otherwise making available, alignment or combination, restriction, erasure or destruction, as referred to in Article 4 at 2) GDPR.
* Supervisory Authority: one or more independent public bodies responsible for supervising the application of the GDPR, in order to protect the constitutional rights and fundamental freedoms of natural persons in connection with the Processing of their Personal Data and to facilitate the free traffic of Personal Data inside the Union, as referred to in Article 4 at 21) and Article 51 GDPR. In the Netherlands, this is the Dutch Data Protection Authority (Autoriteit Persoonsgegevens).

Delft University of Technology has developed together with its partners, at considerable expense and effort, a collection of annotated street-level sensor-recordings from a moving vehicle formatted, termed the DD-Pose Dataset, to facilitate research in the area of environment perception for self-driving vehicles, driver analysis, mobile robotics and intelligent transportation systems.

Delft University of Technology is willing to license the use of the DD-Pose Dataset pursuant to the terms of this Research Use License ("License") to you only if

* you are a Qualified User of the DD-Pose Dataset
* you obtained the DD-Pose Dataset directly from Delft University of Technology, and
* you accept all of the terms contained in this License.

Exercising the rights granted to you below constitutes your agreement to be bound by the terms of this License. The term "you" and "your" means the individual using the DD-Pose Dataset, or clicking "accept" or "agree" or otherwise demonstrating acceptance of this License, and thereby becoming bound by it, and the organization or other legal entity represented by such individual or on whose behalf such individual acts, and all affiliates thereto.

#### 1. Research Use License Grant

(a) Subject to and conditioned upon your compliance with the obligations of this License, Delft University of Technology grants to you a worldwide, royalty-free, non-sublicensable, non-transferable, and non-exclusive copyright and database rights license, solely in furtherance of your Research Purpose, to download, reproduce and use the DD-Pose Dataset.

Delft University of Technology grants no rights to and you shall not in whole or in part: (1) distribute the DD-Pose Dataset, or (2) create and distribute Derivatives that contain or allow the recovery of sensor data of the DD-Pose Dataset, or (3) create and distribute Derivatives that contain or allow the recovery of (existing or new) annotations from the DD-Pose Test Dataset, or (4) create and distribute Derivatives that contain or allow the recovery of Personal Data from the DD-Pose Dataset, or (5) incorporate the DD-Pose Dataset or Derivatives in any publicly available product, service, software or technology for non-Research Purpose.

Delft University of Technology grants you the right to create and distribute Derivatives that consist of abstract representations (e.g. models such as parameters of a trained neural network) derived from the DD-Pose Dataset provided that these Derivatives a) do not involve data derived from the DD-Pose Test Dataset b) are used for Research Purpose and c) are provided free of charge and d) do not involve Personal Data. Delft University of Technology grants you the right to display small samples of the DD-Pose Dataset for the purpose of demonstrating the outcome of your Research Purpose, provided that these samples do not contain Personal Data (e.g. by use of proper anonymization techniques such as blurring of image regions containing identifiable faces and/or license plates).

(b) You shall not use the DD-Pose Dataset for any non-Research Purpose except as separately agreed with Delft University of Technology in a signed writing. Such activity is not licensed or authorized under this License, and if undertaken may result in pursuit of all available remedies, including those for intellectual property rights (including copyright and database rights) infringement, the availability of which you hereby acknowledge.

(c) The DD-Pose Dataset constitute Delft University of Technology's confidential information protected by intellectual property laws and treaties, and is licensed, not sold. All rights not expressly granted in this section 1 are reserved to Delft University of Technology. No other right or license will be implied by conduct or otherwise. Delft University of Technology and/or its partners retain all right, title and interest in the DD-Pose Dataset and all associated intellectual property rights. Patents, trademark and service mark rights, moral rights (including the right of integrity), publicity, privacy, and/or other similar personality rights are not licensed under this License.

(d) You shall not remove, destroy, deface or otherwise alter any legends, notices, statements or marks indicating Delft University of Technology's ownership or the restrictions contained in this License.

(e) You shall not disclose or cause to be disclosed in whole or in part the DD-Pose Dataset or any information contained therein ("Confidential Information") to any third party or use Confidential Information except as specifically authorized by this License. You shall protect the confidentiality of Confidential Information with the same degree of care, but no less than reasonable care, as you use to protect your own confidential information of like nature. Reasonable care includes taking appropriate measures for physical and electronical (e.g. password protection, encryption) access control for the DD-Pose Dataset copy you obtained.

 (f) You shall notify Delft University of Technology immediately when you become aware of any breach or violation of this License, or of any potential or actual infringement of intellectual property rights related to the DD-Pose Dataset.

#### 2. Attribution

In a manner reasonably satisfactory to Delft University of Technology, you shall cause to be conspicuously displayed in all public materials and websites referencing research conducted with the DD-Pose Dataset a notice that the research was conducted using the DD-Pose Dataset. For scientific publications, the preferred DD-Pose Dataset publication is to be cited, as listed on the DD-Pose Dataset website. For other public materials and websites the preferred DD-Pose Dataset publication is to be cited, as listed on the DD-Pose Dataset website, or a link to the DD-Pose Dataset website is to be provided.

#### 3. Feedback

You agree that Delft University of Technology may freely use and exploit in perpetuity any feedback, requirements, recommendations, ideas, bug fixes, reviews, ratings, benchmarks, comments, suggestions, or improvements, that you, may at any time disclose or submit to Delft University of Technology relating to the DD-Pose Dataset for Delft University of Technology's valorization purposes, including for product licensing, support and development, without any obligation or payment to you.

#### 4. Data

(a)  Data distributed with the DD-Pose Dataset

The DD-Pose Dataset contains Personal Data related to identifiable persons (e.g. faces or license plates in image data).

You shall provide Delft University of Technology with all necessary assistance and cooperation in complying with the obligations borne on the basis of the GDPR and other Applicable Legislation and Regulations concerning the Processing of Personal Data. In particular, you shall provide Delft University of Technology with assistance in any event in respect of: Protection of Personal Data, compliance with requests from the Supervisory Authority or another public body; compliance with requests from Data Subjects, reporting Personal Data Breaches.

You shall take appropriate technical and organizational measures to safeguard a level of security attuned to the risk, so that the Processing complies with the requirements under the GDPR and other Applicable Legislation and Regulations concerning the Processing of Personal Data, and the protection of the rights of Data Subjects is safeguarded. In the assessment of the appropriate level of security, you shall take into account the state of the art, the costs of execution, as well as the nature, scope, context and the processing objectives, and the risks varying in terms of probability and seriousness to the rights and freedoms of individuals, especially as a result of the accidental or unlawful destruction, loss, alteration or unauthorized provision of or unauthorized access to data that is transferred, stored or otherwise processed.

Without unreasonable delay and no later than within 24 hours after discovery, you shall notify the Delft University of Technology of a Personal Data Breach or a reasonable suspicion of a Personal Data Breach. You warrant that the information provided is complete, correct and accurate.

(b) Data concerning or arising from DD-Pose Dataset use

You hereby consent to Delft University of Technology's collection and use of your Personal Data (including name, affiliation, email address, meta-data, analytical, diagnostic and technical data, and usage statistics) concerning or arising from your use of the DD-Pose Dataset in order to provide the functionality of and improvement of the DD-Pose Dataset, for product development and marketing purposes, to protect against spam and malware, and for verifying License compliance. You may revoke this consent at any time by giving Delft University of Technology a written notice, at which point your License automatically terminates (see Section 7).

#### 5. Updates

The DD-Pose Dataset may update automatically. You agree to accept such updates subject to this License unless other terms accompany the updates. If so, those other terms will apply. Delft University of Technology is not obligated to make any updates available, to continue making the DD-Pose Dataset available for any period of time, or to provide any technical support of any kind.

#### 6. Compliance

You shall comply with all applicable laws, rules, and regulations in respect of the use of the DD-Pose Dataset. In particular, you shall comply with GDPR, even if you are located outside the territory of the EU.

#### 7. Termination

Without prejudice to any other Delft University of Technology rights or remedies, this License terminates three years after it was granted. In reasonable time before the termination date, a license renewal request can be made on the DD-Pose website. In the event of termination of the license without renewal, you shall immediately destroy all copies and cease all use of the DD-Pose Dataset.    

Without prejudice to any other Delft University of Technology rights or remedies, this License will automatically terminate if you fail to comply with its terms (this includes, among others, if you no longer are a Qualified User of the DD-Pose Dataset) or if you revoke your consent that Delft University of Technology may use your Personal data as specified in Section 4 (b). In such event, you shall immediately destroy all copies and cease all use of the DD-Pose Dataset.

You acknowledge and agree that breach of this License, or any unauthorized use or distribution of the DD-Pose Dataset, would cause irreparable harm to Delft University of Technology, the extent of which would be difficult to ascertain, and that Delft University of Technology will be entitled to seek without limitation immediate injunctive relief in any court of competent jurisdiction under the applicable laws thereof (and Delft University of Technology's right to do so is not subject to arbitration). Delft University of Technology may terminate this License, in whole or in part, upon written notice in the event the DD-Pose Dataset, in whole or in part, is in Delft University of Technology's sole judgment subject to a claim of intellectual property rights infringement, violation of law or rights, or other claim impeding Delft University of Technology's right or ability to grant the licensed rights hereunder.

#### 8. Disclaimer

You acknowledge the DD-Pose Dataset may have inaccuracies, defects or deficiencies that may not be corrected by Delft University of Technology and/or its partners. To the maximum extent permitted by applicable law, Delft University of Technology and/or its partners provide the DD-Pose Dataset "as is" and with all faults, and disclaim all representations, warranties, and conditions, whether express, implied or statutory, including but not limited to representations, warranties and conditions related to: title, non-infringement, merchantability, fitness for a particular purpose, accuracy or completeness, lack of defects, negligence or workmanlike effort, or correspondence to description. The entire risk arising out of use or performance of the DD-Pose Dataset remains with you.

#### 9. Limitation of Liability

EXCLUDING LOSS FOR WHICH DELFT UNIVERSITY OF TECHNOLOGY CANNOT LAWFULLY EXCLUDE LIABILITY, DELFT UNIVERSITY OF TECHNOLOGY WILL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES, OR COST OF PROCUREMENT OF SUBSTITUTE SERVICES, OR DAMAGES FOR LOSS OF REVENUE, PROFITS, GOODWILL, USE, DATA OR OTHER INTANGIBLE LOSSES (EVEN IF DELFT UNIVERSITY OF TECHNOLOGY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES), RESULTING FROM THE USE OF OR INABILITY TO USE THE DD-Pose DATASET OR THE EXERCISE OF THE LICENSED RIGHTS GRANTED HEREUNDER.

#### 10. Severability; Waiver; No Agency; Entire Agreement

If any of the provisions of this License are held to be in violation of applicable law, void, or unenforceable in any jurisdiction, such provisions will not affect the validity of the balance of the License, and such provisions are herewith waived or reformed to the extent necessary for the License to be otherwise enforceable in such jurisdiction. No waiver of any provision of this License will be deemed a further waiver or continuing waiver of such provision or any other provision, and Delft University of Technology's failure to assert any right or provision under this License will not constitute a waiver of such right or provision. This License is the parties' entire agreement relating to its subject matter and supersedes all prior or contemporaneous oral or written communications, proposals, negotiations, understandings, representations and warranties and prevails over any conflicting or additional terms of any quote, order or other communication between the parties relating to its subject matter.

#### 11. Assignment

This License may and shall not be assigned or transferred, or its rights or obligations assigned or delegated, by you, in whole or in part, without the prior written consent of Delft University of Technology. Any assignment made in violation of this section will be void. Delft University of Technology may freely assign this License without the necessity for consent.

#### 12. Governing Law

Dutch law will apply to this Data Processing Agreement. Disputes about this Agreement will be presented to the District Court of The Hague in The Netherlands.

#### 13. Contact Data

The contact data of Delft University of Technology with respect to this License is:

Department of Cognitive Robotics  
Faculty of Mechanical, Maritime and Materials Engineering (3mE)  
Mekelweg 2, 2628 CD Delft, The Netherlands  

*end*
