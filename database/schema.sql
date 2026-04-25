-- ============================================================
-- MIST AI - SRM Student Assistant
-- PostgreSQL Database Schema
-- ============================================================

-- Enable trigram extension for fuzzy text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- 1. PORTALS
-- ============================================================
CREATE TABLE IF NOT EXISTS portals (
    portal_id   SERIAL PRIMARY KEY,
    portal_name TEXT NOT NULL UNIQUE,
    link        TEXT,
    managed_by  TEXT
);

-- ============================================================
-- 2. PORTAL_FEATURES
-- ============================================================
CREATE TABLE IF NOT EXISTS portal_features (
    feature_id   SERIAL PRIMARY KEY,
    portal_id    INT REFERENCES portals(portal_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    description  TEXT,
    UNIQUE(portal_id, feature_name)
);

-- ============================================================
-- 3. PYQ_RESOURCES
-- ============================================================
CREATE TABLE IF NOT EXISTS pyq_resources (
    pyq_id       SERIAL PRIMARY KEY,
    subject_name TEXT NOT NULL,
    subject_code TEXT,
    pyq_link     TEXT,
    source       TEXT,
    UNIQUE(subject_name, subject_code)
);

-- ============================================================
-- 4. UNIVERSITY_INFO
-- ============================================================
CREATE TABLE IF NOT EXISTS university_info (
    info_id    SERIAL PRIMARY KEY,
    topic      TEXT NOT NULL,
    question   TEXT UNIQUE,
    answer     TEXT,
    source_url TEXT
);

-- ============================================================
-- 5. LOCATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS locations (
    location_id SERIAL PRIMARY KEY,
    place_name  TEXT NOT NULL UNIQUE,
    description TEXT
);

-- ============================================================
-- 6. WEBSITE_CONTENT
-- ============================================================
CREATE TABLE IF NOT EXISTS website_content (
    content_id SERIAL PRIMARY KEY,
    page_title TEXT,
    url        TEXT UNIQUE,
    content    TEXT
);

-- ============================================================
-- 7. CHAT_LOGS
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_logs (
    chat_id   SERIAL PRIMARY KEY,
    question  TEXT,
    response  TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- INDEXES for full-text search
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_website_content_fts
    ON website_content USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_university_info_answer_fts
    ON university_info USING GIN (to_tsvector('english', answer));

CREATE INDEX IF NOT EXISTS idx_university_info_question_fts
    ON university_info USING GIN (to_tsvector('english', question));

CREATE INDEX IF NOT EXISTS idx_portal_features_desc_fts
    ON portal_features USING GIN (to_tsvector('english', description));

CREATE INDEX IF NOT EXISTS idx_locations_desc_fts
    ON locations USING GIN (to_tsvector('english', description));


-- ============================================================
-- SEED DATA (ON CONFLICT DO NOTHING prevents duplicates)
-- ============================================================

-- Portals
INSERT INTO portals (portal_name, link, managed_by) VALUES
    ('Academia',        'https://academia.srmist.edu.in',  'SRM IT Department'),
    ('Student Portal',  'https://sp.srmist.edu.in',        'SRM IT Department'),
    ('SRM Website',     'https://www.srmist.edu.in',       'SRM Administration'),
    ('Exam Portal',     'https://examcell.srmist.edu.in',  'SRM Exam Cell'),
    ('Library Portal',  'https://library.srmist.edu.in',   'SRM Library')
ON CONFLICT (portal_name) DO NOTHING;

-- Portal Features
INSERT INTO portal_features (portal_id, feature_name, description) VALUES
    ((SELECT portal_id FROM portals WHERE portal_name='Academia'),       'Attendance',           'View your daily and cumulative attendance records on Academia.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Academia'),       'Timetable',            'Check your class timetable and scheduled sessions on Academia.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Academia'),       'Internal Marks',       'View internal assessment marks and grades on Academia.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Academia'),       'Course Registration',  'Register for courses each semester through Academia.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Student Portal'), 'Fee Payment',          'Pay your tuition and hostel fees through the Student Portal at https://sp.srmist.edu.in.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Student Portal'), 'Scholarship Renewal',  'Apply for and renew scholarships through the Student Portal.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Student Portal'), 'Transport Booking',    'Book university transport and bus pass through the Student Portal.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Student Portal'), 'Hostel Allocation',    'View hostel room allocation details on the Student Portal.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Exam Portal'),    'Exam Hall Ticket',     'Download your examination hall ticket from the Exam Portal at https://examcell.srmist.edu.in.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Exam Portal'),    'Exam Results',         'Check your semester exam results on the Exam Portal.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Exam Portal'),    'Exam Schedule',        'View the upcoming exam timetable on the Exam Portal.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Library Portal'), 'Book Search',          'Search for books available in the SRM library catalog.'),
    ((SELECT portal_id FROM portals WHERE portal_name='Library Portal'), 'E-Resources',          'Access e-books, journals, and digital resources through the Library Portal.')
ON CONFLICT (portal_id, feature_name) DO NOTHING;

-- PYQ Resources
INSERT INTO pyq_resources (subject_name, subject_code, pyq_link, source) VALUES
    ('Database Management Systems',     'CS1502', 'https://thehelpers.vercel.app/dbms',   'The Helpers'),
    ('Data Structures and Algorithms',  'CS1301', 'https://thehelpers.vercel.app/dsa',    'The Helpers'),
    ('Operating Systems',               'CS1401', 'https://thehelpers.vercel.app/os',     'The Helpers'),
    ('Computer Networks',               'CS1503', 'https://thehelpers.vercel.app/cn',     'The Helpers'),
    ('Object Oriented Programming',     'CS1201', 'https://thehelpers.vercel.app/oop',    'The Helpers'),
    ('Discrete Mathematics',            'MA1301', 'https://thehelpers.vercel.app/dm',     'The Helpers'),
    ('Software Engineering',            'CS1601', 'https://thehelpers.vercel.app/se',     'The Helpers'),
    ('Artificial Intelligence',         'CS1701', 'https://thehelpers.vercel.app/ai',     'The Helpers'),
    ('Machine Learning',                'CS1702', 'https://thehelpers.vercel.app/ml',     'The Helpers'),
    ('Theory of Computation',           'CS1504', 'https://thehelpers.vercel.app/toc',    'The Helpers')
ON CONFLICT (subject_name, subject_code) DO NOTHING;

-- University Info
INSERT INTO university_info (topic, question, answer, source_url) VALUES
    ('hostel',        'Does SRM provide hostel?',
     'Yes, SRM Institute of Science and Technology provides hostel facilities for both boys and girls. Hostels include furnished rooms, mess facilities, Wi-Fi, and 24/7 security. You can apply for hostel through the Student Portal.',
     'https://www.srmist.edu.in/hostel'),

    ('hostel',        'What are the hostel fees?',
     'Hostel fees vary depending on the type of room (single, double, or triple sharing). You can check the current fee structure and make payments through the Student Portal at https://sp.srmist.edu.in.',
     'https://sp.srmist.edu.in'),

    ('transport',     'Does SRM provide transport?',
     'Yes, SRM provides bus transport facilities for day scholars. You can book a bus pass and view routes through the Student Portal. The transport service covers major areas around the university.',
     'https://www.srmist.edu.in/transport'),

    ('scholarships',  'What scholarships are available at SRM?',
     'SRM offers merit-based scholarships, founder''s scholarships, sports scholarships, and need-based financial aid. You can apply and renew scholarships through the Student Portal. Details are available on the SRM website.',
     'https://www.srmist.edu.in/scholarships'),

    ('scholarships',  'How do I renew my scholarship?',
     'Scholarship renewal can be done through the Student Portal at https://sp.srmist.edu.in. Log in with your credentials, go to the Scholarship section, and follow the renewal process. Maintain the required GPA to stay eligible.',
     'https://sp.srmist.edu.in'),

    ('departments',   'What departments are in SRM?',
     'SRM has multiple departments including Computer Science, Electronics and Communication, Mechanical, Civil, Electrical, Biomedical, Biotechnology, and more across various schools like the School of Computing, School of Science and Humanities, and School of Engineering.',
     'https://www.srmist.edu.in/departments'),

    ('library',       'Where is the SRM library?',
     'The SRM Central Library is located in the main campus. It offers a vast collection of books, journals, e-resources, and study spaces. You can search for books and access digital resources through the Library Portal at https://library.srmist.edu.in.',
     'https://library.srmist.edu.in'),

    ('library',       'What are the library timings?',
     'The SRM Central Library is generally open from 8:00 AM to 10:00 PM on weekdays and 9:00 AM to 5:00 PM on weekends. Timings may vary during exam periods. Check the Library Portal for the latest schedule.',
     'https://library.srmist.edu.in'),

    ('placement',     'Where is the placement office?',
     'The SRM Placement Office (Career Centre) is located in the main campus building. It handles campus placements, internships, and career guidance. You can contact them through the SRM website or visit in person for placement-related queries.',
     'https://www.srmist.edu.in/placements'),

    ('placement',     'How to register for placements?',
     'Placement registration is done through the Career Centre. Students need to register on the placement portal, upload their resume, and meet eligibility criteria (minimum attendance, no backlogs). Contact the Placement Office for the registration link.',
     'https://www.srmist.edu.in/placements'),

    ('fees',          'Where do I pay fees?',
     'You can pay your tuition fees, hostel fees, and other charges through the Student Portal at https://sp.srmist.edu.in. The portal supports online payment via net banking, debit/credit card, and UPI.',
     'https://sp.srmist.edu.in'),

    ('fees',          'What is the fee structure?',
     'The fee structure varies by program and department. You can view the detailed fee structure on the SRM website or the Student Portal. For specific queries, contact the Accounts Department.',
     'https://www.srmist.edu.in/fees'),

    ('attendance',    'Where can I check attendance?',
     'You can check your attendance on Academia at https://academia.srmist.edu.in. Log in with your credentials and navigate to the Attendance section to view your daily and cumulative attendance percentage.',
     'https://academia.srmist.edu.in'),

    ('attendance',    'What is the minimum attendance requirement?',
     'SRM requires a minimum of 75% attendance in each subject to be eligible for the end-semester exam. Students below 75% may face detention or exam debarment. Check your attendance regularly on Academia.',
     'https://academia.srmist.edu.in'),

    ('exam',          'How can I download hall ticket?',
     'You can download your exam hall ticket from the Exam Portal at https://examcell.srmist.edu.in. Log in with your student credentials, go to the Hall Ticket section, and download it in PDF format.',
     'https://examcell.srmist.edu.in'),

    ('exam',          'When are the semester exams?',
     'Semester exam schedules are published on the Exam Portal at https://examcell.srmist.edu.in. You can also check Academia for updates. The schedule is usually announced a few weeks before examinations.',
     'https://examcell.srmist.edu.in'),

    ('campus',        'What facilities are on campus?',
     'SRM campus has facilities including a central library, multiple food courts, sports complex, gymnasium, health center, ATMs, stationery shops, tech park for innovation, auditoriums, and Wi-Fi across the campus.',
     'https://www.srmist.edu.in'),

    ('admission',     'How to apply for admission at SRM?',
     'You can apply for admission through the SRMJEEE (SRM Joint Engineering Entrance Examination) portal. Visit the SRM website for application details, eligibility criteria, and important dates.',
     'https://www.srmist.edu.in/admissions'),

    ('clubs',         'What student clubs are at SRM?',
     'SRM has a vibrant student life with clubs like Aaruush (tech fest), Milan (cultural fest), coding clubs, robotics clubs, literary clubs, music and dance clubs, photography clubs, and many more departmental clubs.',
     'https://www.srmist.edu.in/student-life')
ON CONFLICT (question) DO NOTHING;

-- Locations
INSERT INTO locations (place_name, description) VALUES
    ('Central Library',     'The SRM Central Library is the main library with a vast collection of books, journals, and digital resources. Located in the main campus.'),
    ('Tech Park',           'The SRM Tech Park is a hub for innovation, startups, and industry collaboration. It houses research labs and incubation centers.'),
    ('Food Court',          'Multiple food courts are available across the campus offering a variety of cuisines including vegetarian and non-vegetarian options.'),
    ('Main Building',       'The Main Administrative Building houses the university offices, admissions, accounts department, and administrative staff.'),
    ('Sports Complex',      'The Sports Complex includes indoor and outdoor facilities for cricket, football, basketball, tennis, swimming, and gymnasium.'),
    ('Health Center',       'The campus Health Center provides medical consultations, first aid, and basic health services for students and staff.'),
    ('Placement Office',    'The Career Centre / Placement Office handles campus recruitments, internships, and career guidance for students.'),
    ('University Building', 'The main university building (UB) houses classrooms, lecture halls, and department offices.'),
    ('Auditorium',          'The T.P. Ganesan Auditorium is the main venue for university events, seminars, and cultural programs.'),
    ('Hostel Block',        'Hostel blocks for boys and girls with furnished rooms, mess facilities, Wi-Fi, and security.')
ON CONFLICT (place_name) DO NOTHING;

-- ============================================================
-- END OF SCHEMA
-- ============================================================
